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