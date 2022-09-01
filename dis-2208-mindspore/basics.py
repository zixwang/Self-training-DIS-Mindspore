import os
import time
import numpy as np
from skimage import io
from PIL import Image
import mindspore
import mindspore.ops as ops

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
    sumError = ops.ReduceSum()(ops.Abs()(ops.Sub()(pred.float(), gt.float())))
    maeError = mindspore.numpy.true_divide(sumError, float(h)*float(w)*255.0+1e-4)

    return maeError


def f1score_torch(pd, gt):

    gtNum = ops.ReduceSum()((gt > 128).float()*1)  # number of ground truth pixels

    pp = pd[gt > 128]
    nn = pd[gt <= 128]

    pp_hist = ops.HistogramFixedWidth(255)(pp, [0,255])
    nn_hist = ops.HistogramFixedWidth(255)(nn, [0,255])

    pp_hist_flip = mindspore.numpy.flipud(pp_hist)
    nn_hist_flip = mindspore.numpy.flipud(nn_hist)

    pp_hist_flip_cum = ops.CumSum()(pp_hist_flip, dim=0)
    nn_hist_flip_cum = ops.CumSum()(nn_hist_flip, dim=0)

    precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
    recall = pp_hist_flip_cum / (gtNum + 1e-4)
    f1 = (1+0.3)*precision*recall/(0.3*precision+recall + 1e-4)

    return ops.Reshape()(precision, (1, precision.shape[0])), ops.Reshape()(recall, (1, recall.shape[0])), ops.Reshape(f1, (1, f1.shape[0]))


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
    sumError = ops.ReduceSum()(ops.Abs()(ops.Sub()(pred.float(), gt.float())))
    maeError = mindspore.numpy.true_divide(sumError, float(h)*float(w)*255.0+1e-4)

    return maeError


def f1score_torch_bear(pd, gt):
    import time
    start = time.time()
    # print(gt.shape)
    gtNum = ops.ReduceSum()((gt > 128).float()*1.0)  # number of ground truth pixels

    pp = pd[gt > 128]  # TP
    nn = pd[gt <= 128]  # FP

    pp_hist = ops.HistogramFixedWidth(255)(pp.float(), [0,255])
    nn_hist = ops.HistogramFixedWidth(255)(nn.float(), [0,255])

    pp_hist_flip = mindspore.numpy.flipud(pp_hist)
    nn_hist_flip = mindspore.numpy.flipud(nn_hist)

    pp_hist_flip_cum = ops.CumSum()(pp_hist_flip, dim=0)
    nn_hist_flip_cum = ops.CumSum()(nn_hist_flip, dim=0)

    precision = (pp_hist_flip_cum)/(pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
    recall = mindspore.numpy.true_divide(pp_hist_flip_cum, (gtNum + 1e-4))
    f1 = (1+0.3)*precision*recall/(0.3*precision+recall + 1e-4)

    print("time for eval: ", time.time()-start)
    return ops.Reshape()(precision, (1, precision.shape[0])), ops.Reshape()(recall, (1, recall.shape[0])), ops.Reshape()(f1, (1, f1.shape[0]))


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