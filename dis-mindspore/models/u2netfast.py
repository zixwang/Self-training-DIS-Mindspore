import mindspore
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