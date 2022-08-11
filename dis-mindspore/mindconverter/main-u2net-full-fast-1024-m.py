import os
from pyexpat import model
from PIL import Image
import time
import numpy as np
import gc
from copy import deepcopy
import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import logging
from mindspore.dataset import GeneratorDataset
from skimage import io
from dataset.semi import SemiDataset
from basics import f1_mae_torch
from models import *
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
mindspore.set_context(device_target="Ascend")

# <====================== (2) 主函数 ======================>
def main(hypar):
    if not os.path.exists(hypar['save_path']):
        os.makedirs(hypar['save_path'])
    if not os.path.exists(hypar['pseudo_mask_path']):
        os.makedirs(hypar['pseudo_mask_path'])

    valset = SemiDataset(root=hypar['test_data_root'], mode='val', size=1024)
    valloader = GeneratorDataset(valset, batch_size=1, shuffle=False,
                           pin_memory=True, num_workers=1, drop_last=False)

    # <====================== (3) 设定训练集，验证集，监督学习 ======================>
    print('\n\n================> Total stage 1/3: Supervised training on labeled images (SupOnly)')

    global MODE
    MODE = 'train'

    trainset = SemiDataset(root=hypar['train_data_root'], mode=MODE, size=1024,
                           labeled_id_path=hypar['labeled_id_path'])
    trainloader = GeneratorDataset(trainset, batch_size=8, shuffle=True,
                             pin_memory=True, num_workers=4, drop_last=True)

    net, optimizer = define_basic_elems(hypar)

    best_model = train(net, optimizer, trainloader, trainset, valloader, valset, hypar)

    # <====================== (4) 换数据集，打伪标签 ======================>
    print('\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images')

    MODE = 'label'

    dataset = SemiDataset(root=hypar['train_data_root'], mode=MODE, size=1024,
                          labeled_id_path=None, unlabeled_id_path=hypar['unlabeled_id_path'], pseudo_mask_path=None)
    dataloader = GeneratorDataset(dataset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=1, drop_last=False)

    label(best_model, dataloader, hypar)

    # <====================== (5) 换数据集，整合，训练 ======================>
    print('\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')

    MODE = 'semi_train'

    trainset = SemiDataset(root=hypar['train_data_root'], mode=MODE, size=1024,
                           labeled_id_path=hypar['labeled_id_path'],
                           unlabeled_id_path=hypar['unlabeled_id_path'],
                           pseudo_mask_path=hypar['pseudo_mask_path'])
    trainloader = GeneratorDataset(trainset, batch_size=8, shuffle=True,
                             pin_memory=True, num_workers=4, drop_last=True)

    net, optimizer = define_basic_elems(hypar)

    train(net, optimizer, trainloader, trainset, valloader, valset, hypar)

    return


# <====================== (6) 规定参数 ======================>
def define_basic_elems(hypar):
    net = hypar["model"]

    if hypar["restore_model"]:
        print("restore model from:")
        print(hypar["model_path"] + "/" + hypar["restore_model"])
        mindspore.load_param_into_net(net, mindspore.load_checkpoint(
            hypar["model_path"] + "/" + hypar["restore_model"]))

    print("--- define optimizer ---")
    optimizer = mindspore.nn.Adam(net.trainable_params(), lr=1e-3,
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    return net, optimizer


# <====================== (7) 设定训练函数 ======================>
def train(net, optimizer, train_dataloaders, train_datasets, valid_dataloaders, valid_datasets, hypar):
    global best_model
    model_path = hypar["model_path"] if MODE == 'train' else hypar["model_path"] + '_after_labeling'
    model_save_fre = hypar["model_save_fre"]
    max_ite = hypar["max_ite"]
    batch_size_train = hypar["batch_size_train"]
    batch_size_valid = hypar["batch_size_valid"]

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    ite_num = hypar["start_ite"]  # count the toal iteration number
    ite_num4val = 0
    running_loss = 0.0  # count the toal loss
    running_tar_loss = 0.0  # count the target output loss
    last_f1 = [0]

    train_num = len(train_datasets)

    net.train()

    start_last = time.time()
    dis_dataloader = train_dataloaders
    epoch_num = hypar["max_epoch_num"]
    notgood_cnt = 0

    # <====================== (8) 设定训练的epoch ======================>
    for epoch in range(epoch_num):  # set the epoch num as 1000000
        print("\nEpoch Number: " + str(epoch + 1))
        tbar = tqdm(dis_dataloader)
        # <====================== (9) 每个epoch中，进行训练，得到loss ======================>
        for i, data in enumerate(tbar):

            if ite_num >= max_ite:
                print("Training Reached the Maximal Iteration Number ", max_ite)
                exit()

            ite_num, ite_num4val = ite_num + 1, ite_num4val + 1

            # inputs, labels = data['image'], data['label']
            inputs, labels = data
            labels = mindspore.ops.ExpandDims(labels, 1)

            if hypar["model_digit"] == "full":
                inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.FloatTensor)
            else:
                inputs, labels = inputs.type(torch.HalfTensor), labels.type(torch.HalfTensor)

            # if torch.cuda.is_available():
            #     inputs_v, labels_v = inputs.cuda(), labels.cuda()
            # else:
            inputs_v, labels_v = inputs, labels

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()

            # forward + backward + optimize
            # start_inf = time.time()
            ds = net(inputs_v)

            # a = ds[0].cpu().data.numpy()
            # start_loss = time.time()
            # loss2, loss = multi_loss_fusion(ds, labels_v)
            loss2, loss = net.compute_loss(ds, labels_v)
            # "loss2" is computed based on the final output of our model
            # "loss" is the sum of all the outputs including side outputs
            # for dense supervision
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            tbar.set_description('ite: %d, Loss: %.3f' % (ite_num, running_loss / ite_num4val))

            del ds, loss2, loss
            end_inf_loss_back = time.time() - start_inf_loss_back

            logging.info(">>>" + model_path.split('/')[
                -1] + "- [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, "
                      "time_read: %3f" % (
                             epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num,
                             running_loss / ite_num4val,
                             running_tar_loss / ite_num4val, time.time() - start_last,
                             time.time() - start_last - end_inf_loss_back))
            start_last = time.time()

            # <====================== (9) 1000个iteration(或者一些epoch)后，进行验证，得到评价指标值 ======================>
            if ite_num % model_save_fre == 0:  # validate every 1000 iterations
                notgood_cnt += 1
                net.eval()
                tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid(
                    net, valid_dataloaders, valid_datasets, hypar)
                net.train()  # resume train

                # <====================== (10) 若评价指标值是最好的，则更新指标，并保存模型 ======================>
                tmp_out = 0
                print("last_f1:", last_f1)
                print("tmp_f1:", tmp_f1)
                for fi in range(len(last_f1)):
                    if tmp_f1[fi] > last_f1[fi]:
                        tmp_out = 1
                print("tmp_out:", tmp_out)
                logging.info("tmp_out: %s" % tmp_out)
                if tmp_out:
                    notgood_cnt = 0
                    last_f1 = tmp_f1
                    tmp_f1_str = [str(round(f1x, 4)) for f1x in tmp_f1]
                    tmp_mae_str = [str(round(mx, 4)) for mx in tmp_mae]
                    maxf1 = '_'.join(tmp_f1_str)
                    meanM = '_'.join(tmp_mae_str)

                    model_name = "/gpu_itr_" + str(ite_num) + \
                                 "_traLoss_" + str(np.round(running_loss / ite_num4val, 4)) + \
                                 "_traTarLoss_" + str(np.round(running_tar_loss / ite_num4val, 4)) + \
                                 "_valLoss_" + str(np.round(val_loss / (i_val + 1), 4)) + \
                                 "_valTarLoss_" + str(np.round(tar_loss / (i_val + 1), 4)) + \
                                 "_maxF1_" + maxf1 + \
                                 "_mae_" + meanM + \
                                 "_time_" + \
                                 str(np.round(np.mean(np.array(tmp_time)) /
                                              batch_size_valid, 6)) + ".pth"
                    mindspore.save_checkpoint(net.parameters_dict(), model_path + model_name)
                    best_model = deepcopy(net)
                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if notgood_cnt >= hypar["early_stop"]:
                    print("No improvements in the last " + str(notgood_cnt) +
                          " validation periods, so training stopped !")
                    return best_model
                    # exit()

    # <====================== (11) 保存中间的checkpoints，并返回 ======================>

    print("Training Reaches The Maximum Epoch Number")

    return best_model


def valid(net, valid_dataloader, valid_dataset, hypar):
    net.eval()
    print("Validating...")

    val_loss, tar_loss, val_cnt = 0.0, 0.0, 0.0
    tmp_f1, tmp_mae, tmp_time = [], [], []

    # for k in range(len(valid_dataloaders)):

    #     valid_dataloader = valid_dataloaders[k]
    #     valid_dataset = valid_datasets[k]

    val_num = len(valid_dataset)

    mybins = np.arange(0, 256)
    PRE = np.zeros((val_num, len(mybins) - 1))
    REC = np.zeros((val_num, len(mybins) - 1))
    F1 = np.zeros((val_num, len(mybins) - 1))
    MAE = np.zeros(val_num)

    tbar = tqdm(valid_dataloader)
    with torch.no_grad():
        for i_val, data_val in enumerate(tbar):
            val_cnt = val_cnt + 1.0

            inputs_val, labels_val, id_val = data_val
            labels_val = mindspore.ops.ExpandDims(labels_val, 1)
            shapes_val = io.imread(os.path.join(hypar['test_data_root'], id_val[0].split(',')[1])).shape
            imidx_val = i_val
            # imidx_val, inputs_val, labels_val, shapes_val = \
            # data_val['imidx'], data_val['image'], data_val['label'], data_val['shape']

            if hypar["model_digit"] == "full":
                inputs_val = inputs_val.type(torch.FloatTensor)
                labels_val = labels_val.type(torch.FloatTensor)
            else:
                inputs_val = inputs_val.type(torch.HalfTensor)
                labels_val = labels_val.type(torch.HalfTensor)

            # if torch.cuda.is_available():
            #     inputs_val_v, labels_val_v = inputs_val.cuda(), labels_val.cuda()
            # else:
            inputs_val_v, labels_val_v = inputs_val, labels_val

            t_start = time.time()
            ds_val = net(inputs_val_v)
            t_end = time.time() - t_start
            tmp_time.append(t_end)

            loss2_val, loss_val = net.compute_loss(ds_val, labels_val_v)

            # compute F measure
            for t in range(hypar["batch_size_valid"]):
                # i_test = imidx_val[t].data.numpy()
                i_test = i_val
                pred_val = ds_val[0][t, :, :, :]  # B x 1 x H x W

                # recover the prediction spatial size to the original image size
                pred_val = mindspore.ops.Squeeze(mindspore.ops.interpolate(mindspore.ops.ExpandDims(
                    pred_val, 0), sizes=(shapes_val[0], shapes_val[1]), mode='bilinear'))

                ma = mindspore.ops.ArgMaxWithValue(pred_val)
                mi = mindspore.ops.ArgMinWithValue(pred_val)
                pred_val = (pred_val - mi) / (ma - mi)  # max = 1

                gt = np.squeeze(
                    io.imread(os.path.join(hypar['test_data_root'], id_val[0].split(',')[1])))  # max = 255
                with torch.no_grad():
                    gt = torch.tensor(gt) #.cuda()

                pred_val = pred_val * 255
                pre, rec, f1, mae = f1_mae_torch(
                    pred_val, gt, valid_dataset, i_test, mybins, hypar)

                # if hypar["valid_out_dir"] != "":
                #     if not os.path.exists(hypar["valid_out_dir"]):
                #         os.mkdir(hypar["valid_out_dir"])
                #     dataset_folder = hypar["valid_out_dir"]
                #     io.imsave(os.path.join(
                #         dataset_folder, os.path.basename(id_val[0].split(',')[1])), pred_val.cpu().data.numpy().astype(np.uint8))

                PRE[i_test, :] = pre
                REC[i_test, :] = rec
                F1[i_test, :] = f1
                MAE[i_test] = mae

                del ds_val, gt
                gc.collect()
                # torch.cuda.empty_cache()

            val_loss += loss_val.item()  # data[0]
            tar_loss += loss2_val.item()  # data[0]
            tbar.set_description('F1: %f, MAE: %f' % (np.amax(F1[i_test, :]), MAE[i_test]))
            logging.info("[validating: %5d/%5d] val_ls:%f, tar_ls: %f, f1: %f, mae: %f, time: %f" % (i_val, val_num,
                                                                                                     val_loss / (
                                                                                                             i_val + 1),
                                                                                                     tar_loss / (
                                                                                                             i_val + 1),
                                                                                                     np.amax(
                                                                                                         F1[i_test, :]),
                                                                                                     MAE[i_test],
                                                                                                     t_end))

            del loss2_val, loss_val

    print('============================')
    PRE_m = np.mean(PRE, 0)
    REC_m = np.mean(REC, 0)
    f1_m = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)
    tmp_f1.append(np.amax(f1_m))
    tmp_mae.append(np.mean(MAE))
    print("The max F1 Score: %f" % (np.max(f1_m)))
    print("The MAE Score: %f" % (np.mean(MAE)))
    logging.info("The max F1 Score: %f" % (np.max(f1_m)))
    logging.info("The MAE Score: %f" % (np.mean(MAE)))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time


# <====================== (13) 打伪标签 ======================>
def label(net, dataloader, args):
    net.eval()
    with torch.no_grad():
        for img, mask, id in tqdm(dataloader):
            # img = img.cuda()

            shapes_val = io.imread(os.path.join(hypar['train_data_root'], id[0].split(',')[1])).shape

            pred = net(img)[0]

            pred = mindspore.ops.Squeeze(mindspore.ops.interpolate
                                 (pred, sizes=(shapes_val[0], shapes_val[1]), mode='bilinear'))

            ma = mindspore.ops.ArgMaxWithValue(pred)
            mi = mindspore.ops.ArgMinWithValue(pred)
            pred = (pred - mi) / (ma - mi)  # max = 1
            pred = pred * 255

            pred = Image.fromarray(pred.squeeze(0).cpu().numpy().astype(np.uint8))
            pred.save('%s/%s' % (hypar['pseudo_mask_path'], os.path.basename(id[0].split(',')[1])))
            logging.info('Label: %s/%s' % (hypar['pseudo_mask_path'], os.path.basename(id[0].split(',')[1])))


# <====================== main ======================>
if __name__ == "__main__":

    # 0: overfitting dataset_6, 1: train and valid on dataset_4 and dataset_5, respectively
    overfit_flag = 1
    output_valid = 1

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%d %b %Y, %H:%M:%S',
                        filename='log/runninglog.log',
                        filemode='w')

    # --- hyperparamters of training ---
    hypar = {"cache_size": [1024, 1024], "cache_boost_train": False, "cache_boost_valid": False,
             "input_size": [1024, 1024], "crop_size": [1024, 1024], "random_flip_h": 1, "random_flip_v": 0}

    # cache data spatial size
    # cache_boost==True: we will loaded the cached .pt file
    # for the whole dataset into RAM and will further reduce the image loading time
    # mdoel input spatial size

    # --- data augmentation parameters ---
    # random crop size from the input

    print("building model...")
    hypar["model"] = U2NETFAST()
    hypar["model_name"] = "U2NET_full_fast_1024"
    print("Model Name: ", hypar["model_name"])

    # stop the training when no improvement in the past 100 validation periods
    hypar["early_stop"] = 20  # 100
    # validate and output the model weights every 1000 iterations
    hypar["model_save_fre"] = 2000  # 1000
    hypar["restore_model"] = ""
    # "gpu_itr_106000_traLoss_0.1989_traTarLoss_0.0091_valLoss_4.7497
    # _valTarLoss_0.6625_maxF1_0.8198_mae_0.0731_time_0.04986.pth"
    # "gpu_itr_33000_traLoss_0.0459_traTarLoss_0.0459_valLoss_0.3645
    # _valTarLoss_0.3645_maxF1_0.7142_mae_0.1132_time_0.005121.pth"
    if hypar["restore_model"] != "":
        hypar["start_ite"] = int(hypar["restore_model"].split("_")[2])

    # batch size for training and validation process
    hypar["batch_size_train"] = 8
    hypar["batch_size_valid"] = 1
    print("batch size: ", hypar["batch_size_train"])

    # train or testing mode
    hypar["mode"] = "train"  # "valid"
    hypar["model_digit"] = "full"  # "half"

    # maximum iteration number
    hypar["max_ite"] = 10000000
    hypar["max_epoch_num"] = 1000000
    hypar["start_ite"] = 0

    dataset_4 = {"name": "GOS-TR",
                 "im_dir": "DIS/train/im",
                 "gt_dir": "DIS/train/gt",
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "cache_dir": "DIS/train/cache"}

    dataset_5 = {"name": "GOS-TE",
                 "im_dir": "DIS/test/im",
                 "gt_dir": "DIS/test/gt",
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "cache_dir": "DIS/test/cache"}

    dataset_6 = {"name": "GOS-TE-01",
                 "im_dir": "DIS/test/im_test_01",
                 "gt_dir": "DIS/test/gt_test_01",
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "cache_dir": "DIS/test/cache_test_01"}

    hypar['save_path'] = 'outdir/models/1_4'
    hypar["model_path"] = os.path.join(hypar['save_path'], 'saved_models_dis')
    hypar['pseudo_mask_path'] = 'outdir/pseudo_masks/1_4'
    hypar['labeled_id_path'] = 'dataset/splits/1_4/labeled.txt'
    hypar['unlabeled_id_path'] = 'dataset/splits/1_4/unlabeled.txt'

    hypar['train_data_root'] = 'DIS/train'
    hypar['test_data_root'] = 'DIS/test'

    # output valid results or not
    hypar["valid_out_dir"] = hypar["model_path"]
    if output_valid == 0:
        hypar["valid_out_dir"] = ""

    main(hypar)

# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================

import os
import time
import torch
import numpy as np
from skimage import io
from PIL import Image


def f1score(pd, gt, mybins):

    gtNum = gt[gt > 128].size

    pp = pd[gt > 128]
    nn = pd[gt <= 128]

    pp_hist, pp_edges = np.histogram(pp, bins=mybins)
    nn_hist, nn_edges = np.histogram(nn, bins=mybins)

    pp_hist_flip = np.flipud(pp_hist)
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip)
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)

    precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)

    recall = pp_hist_flip_cum / (gtNum + 1e-4)
    f1 = (1+0.3)*precision*recall/(0.3*precision+recall + 1e-4)

    return np.reshape(precision, (1, len(precision))), np.reshape(recall, (1, len(recall))), np.reshape(f1, (1, len(f1)))


def PRF1Scores(d, lbl_name_list, imidx_val, d_dir, mybins):

    predict = d
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')

    i_test = imidx_val.data.numpy()[0]
    gt = io.imread(lbl_name_list[i_test[0]])
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]
    imo = im.resize((gt.shape[1], gt.shape[0]), resample=Image.BILINEAR)
    pb_np = np.array(imo)

    pb_np255 = (pb_np[:, :, 0]-np.amin(pb_np[:, :, 0])) / \
        (np.amax(pb_np[:, :, 0])-np.amin(pb_np[:, :, 0]))*255
    pre, rec, f1 = f1score(pb_np255, gt, mybins)
    mae = compute_mae(pb_np255, gt)

    return pre, rec, f1, mae


def PRF1ScoresFastNpy(d, val_folder_path, lbl_name_list, imidx_val, d_dir, mybins):

    predict = d
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')

    i_test = imidx_val.data.numpy()[0]
    gt = io.imread(val_folder_path+lbl_name_list[i_test][0]+'.png')
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    imo = im.resize((gt.shape[1], gt.shape[0]), resample=Image.BILINEAR)
    pb_np = np.array(imo)
    pb_np255 = (pb_np[:, :, 0]-np.amin(pb_np[:, :, 0])) / \
        (np.amax(pb_np[:, :, 0])-np.amin(pb_np[:, :, 0]))*255
    pre, rec, f1 = f1score(pb_np255, gt, mybins)
    mae = compute_mae(pb_np255, gt)

    return pre, rec, f1, mae


def GOSPRF1Scores(d, val_folder_path, lbl_name_list, imidx_val, d_dir, mybins):

    predict = d
    predict = predict.squeeze()
    pb_np = predict.cpu().data.numpy()

    i_test = imidx_val.data.numpy()[0]
    gt = io.imread(val_folder_path+lbl_name_list[i_test][0]+'.png')

    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    pb_np255 = (pb_np-np.amin(pb_np))/(np.amax(pb_np)-np.amin(pb_np))*255

    pre, rec, f1 = f1score(pb_np255, gt, mybins)
    mae = compute_mae(pb_np255, gt)

    return pre, rec, f1, mae


def GOSPRF1ScoresCache(pred, gt, valid_dataset, idx, mybins, hypar):

    tic = time.time()

    pb_np = pred.cpu().data.numpy()
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]
    pb_np255 = pb_np*255

    pre, rec, f1 = f1score(pb_np255, gt, mybins)
    print("time for numpy f1: ", time.time()-tic)
    mae = compute_mae(pb_np255, gt)

    if hypar["valid_out_dir"] != "":
        if not os.path.exists(hypar["valid_out_dir"]):
            os.mkdir(hypar["valid_out_dir"])
        dataset_folder = os.path.join(
            hypar["valid_out_dir"], valid_dataset.dataset["data_name"][idx])
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)
        io.imsave(os.path.join(
            dataset_folder, valid_dataset.dataset["im_name"][idx]+".png"), pb_np255.astype(np.uint8))
    print("time for evaluation : ", time.time()-tic)

    return pre, rec, f1, mae


def mae_torch(pred, gt):

    h, w = gt.shape[0:2]
    sumError = mindspore.ops.ReduceSum(mindspore.ops.Abs(mindspore.ops.Sub(pred.float(), gt.float())))
    maeError = mindspore.numpy.true_divide(sumError, float(h)*float(w)*255.0+1e-4)

    return maeError


def f1score_torch(pd, gt):

    gtNum = mindspore.ops.ReduceSum((gt > 128).float()*1)  # number of ground truth pixels

    pp = pd[gt > 128]
    nn = pd[gt <= 128]

    pp_hist = mindspore.ops.HistogramFixedWidth(pp, bins=255, min=0, max=255)
    nn_hist = mindspore.ops.HistogramFixedWidth(nn, bins=255, min=0, max=255)

    pp_hist_flip = mindspore.numpy.flipud(pp_hist)
    nn_hist_flip = mindspore.numpy.flipud(nn_hist)

    pp_hist_flip_cum = mindspore.ops.CumSum(pp_hist_flip, dim=0)
    nn_hist_flip_cum = mindspore.ops.CumSum(nn_hist_flip, dim=0)

    precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
    recall = pp_hist_flip_cum / (gtNum + 1e-4)
    f1 = (1+0.3)*precision*recall/(0.3*precision+recall + 1e-4)

    return mindspore.ops.Reshape(precision, (1, precision.shape[0])), mindspore.ops.Reshape(recall, (1, recall.shape[0])), mindspore.ops.Reshape(f1, (1, f1.shape[0]))


def f1_mae_torch(pred, gt, valid_dataset, idx, mybins, hypar):

    import time
    tic = time.time()

    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    pre, rec, f1 = f1score_torch(pred, gt)
    mae = mae_torch(pred, gt)

    # if hypar["valid_out_dir"] != "":
    #     if not os.path.exists(hypar["valid_out_dir"]):
    #         os.mkdir(hypar["valid_out_dir"])
    #     dataset_folder = os.path.join(
    #         hypar["valid_out_dir"], valid_dataset.dataset["data_name"][idx])
    #     if not os.path.exists(dataset_folder):
    #         os.mkdir(dataset_folder)
    #     io.imsave(os.path.join(
    #         dataset_folder, valid_dataset.dataset["im_name"][idx]+".png"), pred.cpu().data.numpy().astype(np.uint8))
    # print(valid_dataset.dataset["im_name"][idx]+".png")
    # print("time for evaluation : ", time.time()-tic)

    return pre.cpu().data.numpy(), rec.cpu().data.numpy(), f1.cpu().data.numpy(), mae.cpu().data.numpy()


# FOR BEAR COMPUTATION

def mae_torch_bear(pred, gt):

    h, w = gt.shape[0:2]
    sumError = mindspore.ops.ReduceSum(mindspore.ops.Abs(mindspore.ops.Sub(pred.float(), gt.float())))
    maeError = mindspore.numpy.true_divide(sumError, float(h)*float(w)*255.0+1e-4)

    return maeError


def f1score_torch_bear(pd, gt):
    import time
    start = time.time()
    # print(gt.shape)
    gtNum = mindspore.ops.ReduceSum((gt > 128).float()*1.0)  # number of ground truth pixels

    pp = pd[gt > 128]  # TP
    nn = pd[gt <= 128]  # FP

    pp_hist = mindspore.ops.HistogramFixedWidth(pp.float(), bins=255, min=0, max=255)
    nn_hist = mindspore.ops.HistogramFixedWidth(nn.float(), bins=255, min=0, max=255)

    pp_hist_flip = mindspore.numpy.flipud(pp_hist)
    nn_hist_flip = mindspore.numpy.flipud(nn_hist)

    pp_hist_flip_cum = mindspore.ops.CumSum(pp_hist_flip, dim=0)
    nn_hist_flip_cum = mindspore.ops.CumSum(nn_hist_flip, dim=0)

    precision = (pp_hist_flip_cum)/(pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
    recall = mindspore.numpy.true_divide(pp_hist_flip_cum, (gtNum + 1e-4))
    f1 = (1+0.3)*precision*recall/(0.3*precision+recall + 1e-4)

    print("time for eval: ", time.time()-start)
    return mindspore.ops.Reshape(precision, (1, precision.shape[0])), mindspore.ops.Reshape(recall, (1, recall.shape[0])), mindspore.ops.Reshape(f1, (1, f1.shape[0]))


def f1_mae_torch_bear(pred, gt, mybins):

    import time
    tic = time.time()

    if(len(gt.shape) > 2):
        gt = gt[:, :, 0]

    pre, rec, f1 = f1score_torch_bear(pred, gt)
    mae = mae_torch_bear(pred, gt)

    return pre.cpu().data.numpy(), rec.cpu().data.numpy(), f1.cpu().data.numpy(), mae.cpu().data.numpy()


def PRF1Scores_per(d, gt, d_dir, mybins):

    predict = d
    predict = predict.squeeze()
    pb_np = predict.cpu().data.numpy()

    gt = gt.squeeze().cpu().data.numpy()

    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    pb_np255 = (pb_np-np.amin(pb_np))/(np.amax(pb_np)-np.amin(pb_np))*255
    pre, rec, f1 = f1score(pb_np255, gt, mybins)
    mae = compute_mae(pb_np255, gt)

    return pre, rec, f1, mae


def compute_IoU(d, lbl_name_list, imidx_val, d_dir, mybins):
    predict = d
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')

    i_test = imidx_val.data.numpy()[0]
    gt = io.imread(lbl_name_list[i_test[0]])
    if len(gt.shape) > 2:
        gt = gt[:, :, 0]

    imo = im.resize((gt.shape[1], gt.shape[0]), resample=Image.BILINEAR)
    pb_np = np.array(imo)

    pb_np = (pb_np[:, :, 0]+1e-8)/(np.amax(pb_np[:, :, 0])+1e-8)
    gt = (gt+1e-8)/(np.amax(gt)+1e-8)

    pb_bw = pb_np > 0.5
    gt_bw = gt > 0.5

    pb_and_gt = np.logical_and(pb_bw, gt_bw)
    numerator = np.sum(pb_and_gt.astype(np.float))+1e-8
    demoninator = np.sum(pb_bw.astype(np.float)) + \
        np.sum(gt_bw.astype(np.float))-numerator+1e-8

    return numerator/demoninator


def compute_mae(mask1, mask2):
    h, w = mask1.shape
    sumError = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
    maeError = sumError/(float(h)*float(w)*255.0)

    return maeError

# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================

import mindspore.nn as nn
import mindspore.ops.operations as P

bce_loss = nn.BCELoss(reduction='mean')


def muti_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):

        if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:

            tmp_target = mindspore.ops.interpolate(target, sizes=preds[i].size()[
                                                    2:], coordinate_transformation_mode="align_corners", mode='bilinear')
            loss = loss + bce_loss(preds[i], tmp_target)
        else:
            loss = loss + bce_loss(preds[i], target)
        if i == 0:
            loss0 = loss
    return loss0, loss


class REBNCONV(nn.Cell):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, pad_mode='pad', padding=1 * dirate, dilation=1 * dirate, has_bias=True)
        self.bn_s1 = nn.BatchNorm2d(num_features=out_ch, momentum=0.9)
        self.relu_s1 = nn.ReLU()

    def construct(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


# upsample tensor 'src' to have the same spatial size with tensor 'tar'


def _upsample_like(src, tar):
    src = mindspore.ops.interpolate(src, sizes=tar.shape[2:], mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Cell):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(RSU7, self).__init__()

        self.img_size = img_size
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  # 1 -> 1/2

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def construct(self, x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(P.Concat(1)((hx7, hx6)))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(P.Concat(1)((hx6dup, hx5)))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(P.Concat(1)((hx5dup, hx4)))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(P.Concat(1)((hx4dup, hx3)))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(P.Concat(1)((hx3dup, hx2)))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(P.Concat(1)((hx2dup, hx1)))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Cell):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def construct(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(P.Concat(1)((hx6, hx5)))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(P.Concat(1)((hx5dup, hx4)))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(P.Concat(1)((hx4dup, hx3)))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(P.Concat(1)((hx3dup, hx2)))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(P.Concat(1)((hx2dup, hx1)))

        return hx1d + hxin


### RSU-5 ###


class RSU5(nn.Cell):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def construct(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(P.Concat(1)((hx5, hx4)))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(P.Concat(1)((hx4dup, hx3)))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(P.Concat(1)((hx3dup, hx2)))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(P.Concat(1)((hx2dup, hx1)))

        return hx1d + hxin


### RSU-4 ###


class RSU4(nn.Cell):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def construct(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(P.Concat(1)((hx4, hx3)))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(P.Concat(1)((hx3dup, hx2)))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(P.Concat(1)((hx2dup, hx1)))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Cell):  # UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def construct(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(P.Concat(1)((hx4, hx3)))
        hx2d = self.rebnconv2d(P.Concat(1)((hx3d, hx2)))
        hx1d = self.rebnconv1d(P.Concat(1)((hx2d, hx1)))

        return hx1d + hxin


# U^2-Net #### GradLayer
class U2NETFAST(nn.Cell):

    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETFAST, self).__init__()

        self.conv_in = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=True)
        # self.pool_in = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.side2 = nn.Conv2d(in_channels=64, out_channels=out_ch, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.side3 = nn.Conv2d(in_channels=128, out_channels=out_ch, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.side4 = nn.Conv2d(in_channels=256, out_channels=out_ch, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.side5 = nn.Conv2d(in_channels=512, out_channels=out_ch, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.side6 = nn.Conv2d(in_channels=512, out_channels=out_ch, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)

        # self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    @staticmethod
    def compute_loss(preds, targets):
        return muti_loss_fusion(preds, targets)

    def construct(self, x):
        hx = x

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(P.Concat(1)((hx6up, hx5)))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(P.Concat(1)((hx5dup, hx4)))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(P.Concat(1)((hx4dup, hx3)))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(P.Concat(1)((hx3dup, hx2)))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(P.Concat(1)((hx2dup, hx1)))

        # side output
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        return [P.Sigmoid(d1), P.Sigmoid(d2), P.Sigmoid(d3), P.Sigmoid(d4), P.Sigmoid(d5),
                P.Sigmoid(d6)]




# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================




from dataset.transform import crop, hflip, normalize, resize, blur, cutout
import math
import os
from PIL import Image
import random
from mindspore.dataset import GeneratorDataset

class SemiDataset:
    def __init__(self, root, mode, size, labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None):
        """
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.root = root
        self.mode = mode
        self.size = size

        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = \
                self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids
        else:
            if mode == 'val':
                id_path = 'dataset/splits/test.txt'
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(',')[0]))

        if self.mode == 'val' or self.mode == 'label':
            mask = Image.open(os.path.join(self.root, id.split(',')[1]))
            img, mask = resize(img, mask, self.size)
            img, mask = normalize(img, mask)
            return img, mask, id

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.root, id.split(',')[1]))
        else:
            # (self.mode == 'semi_train' and id in self.unlabeled_ids)
            fname = os.path.basename(id.split(',')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # <=============== basic augmentation on all training images ===============>
        img, mask = resize(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # <=============== strong augmentation on unlabeled images ===============>
        if self.mode == 'semi_train' and id in self.unlabeled_ids:
            if random.random() < 0.8:
                img = mindspore.dataset.vision.RandomColorAdjust(0.5, 0.5, 0.5, 0.25)(img)
            img = mindspore.dataset.vision.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = cutout(img, mask, p=0.5)

        img, mask = normalize(img, mask)

        return img, mask

    def __len__(self):
        return len(self.ids)




# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================
# ===============================================================================================



import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import mindspore
from mindspore.dataset import py_transforms as transforms


def crop(img, mask, size):
    # padding height or width if smaller than cropping size
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    # cropping
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
    ])(img)
    if mask is not None:       
        mask = 	mindspore.tensor.from_numpy(np.array(mask)).long()
        mask = mindspore.numpy.true_divide(mask, 255.0)
        return img, mask
    return img


def resize(img, mask, size):
    img = img.resize((size, size), Image.BILINEAR)
    mask = mask.resize((size, size), Image.BILINEAR)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 255

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask
