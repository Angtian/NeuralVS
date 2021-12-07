import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from models.UpsamplingLayer import DoubleConv, Up


vgg_layers = {'pool4': 24, 'pool5': 31}
net_stride = {'vgg_pool4': 16, 'vgg_pool5': 32, 'resnet50': 32, 'resnext50': 32, 'resnetext': 8, 'resnetupsample': 8, 'resnet50_pre': 16}
net_out_dimension = {'vgg_pool4': 512, 'vgg_pool5': 512, 'resnet50': 2048, 'resnext50': 2048, 'resnetext': 256, 'resnetupsample': 2048, 'resnet50_pre': 1024}


class ResnetUpSample(nn.Module):
    def __init__(self, pretrained):
        super(ResnetUpSample, self).__init__()
        net = models.resnet50(pretrained=pretrained)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.extractor = nn.Sequential()
        self.extractor.add_module('0', net.conv1)
        self.extractor.add_module('1', net.bn1)
        self.extractor.add_module('2', net.relu)
        self.extractor.add_module('3', net.maxpool)
        self.extractor.add_module('4', net.layer1)
        self.extractor.add_module('5', net.layer2)
        self.extractor.add_module('6', net.layer3)
        self.extractor.add_module('7', net.layer4)

    def forward(self, x):
        x = self.extractor(x)
        return self.upsample(x)


class ResNetExt(nn.Module):
    def __init__(self, pretrained):
        super(ResNetExt, self).__init__()
        net = models.resnet50(pretrained=pretrained)
        self.extractor = nn.Sequential()
        self.extractor.add_module('0', net.conv1)
        self.extractor.add_module('1', net.bn1)
        self.extractor.add_module('2', net.relu)
        self.extractor.add_module('3', net.maxpool)
        self.extractor.add_module('4', net.layer1)
        self.extractor.add_module('5', net.layer2)
        self.extractor1 = net.layer3
        self.extractor2 = net.layer4

        self.upsample0 = DoubleConv(2048, 1024)
        self.upsample1 = Up(2048, 1024, 512)
        self.upsample2 = Up(1024, 512, 256)

    def forward(self, x):
        x1 = self.extractor(x)
        x2 = self.extractor1(x1)
        x3 = self.extractor2(x2)
        return self.upsample2(self.upsample1(self.upsample0(x3), x2), x1)


def resnet50_pre(pretrain):
    net = models.resnet50(pretrained=pretrain)
    extractor = nn.Sequential()
    extractor.add_module('0', net.conv1)  # add_module(name, module)
    extractor.add_module('1', net.bn1)
    extractor.add_module('2', net.relu)
    extractor.add_module('3', net.maxpool)
    extractor.add_module('4', net.layer1)
    extractor.add_module('5', net.layer2)
    extractor.add_module('6', net.layer3)
    return extractor


def resnetupsample(pretrain):
    net = ResnetUpSample(pretrained=pretrain)
    return net


def resnetext(pretrain):
    net = ResNetExt(pretrained=pretrain)
    return net


def vgg16(layer='pool4'):
    net = models.vgg16(pretrained=True)
    model = nn.Sequential()
    features = nn.Sequential()
    for i in range(0, vgg_layers[layer]):
        features.add_module('{}'.format(i), net.features[i])
    model.add_module('features', features)
    return model


def resnet50(pretrain):
    net = models.resnet50(pretrained=pretrain)
    extractor = nn.Sequential()
    extractor.add_module('0', net.conv1)
    extractor.add_module('1', net.bn1)
    extractor.add_module('2', net.relu)
    extractor.add_module('3', net.maxpool)
    extractor.add_module('4', net.layer1)
    extractor.add_module('5', net.layer2)
    extractor.add_module('6', net.layer3)
    extractor.add_module('7', net.layer4)
    return extractor


# original_img_size = torch.Size([224, 300])
# calculate which patch contains kp. if (1, 1) and line size = 9, return 1*9+1 = 10
def keypoints_to_pixel_index(keypoints, downsample_rate, original_img_size=(480, 640)):
    # line_size = 9
    line_size = original_img_size[1] // downsample_rate
    # round down, new coordinate (keypoints[:,:,0]//downsample_rate, keypoints[:, :, 1] // downsample_rate)
    return keypoints[:, :, 0] // downsample_rate * line_size + keypoints[:, :, 1] // downsample_rate


def get_noise_pixel_index(keypoints, max_size, n_samples, obj_mask=None):
    n = keypoints.shape[0]
    
    # remove the point in keypoints by set probability to 0 otherwise 1 -> mask [n, size] with 0 or 1
    mask = torch.ones((n, max_size), dtype=torch.float32).to(keypoints.device)
    mask = mask.scatter(1, keypoints.type(torch.long), 0.) 
    if obj_mask is not None:
        mask *= obj_mask

    # generate the sample by the probabilities
    return torch.multinomial(mask, n_samples)


class GlobalLocalConverter(nn.Module):
    def __init__(self, local_size):
        super(GlobalLocalConverter, self).__init__()
        self.local_size = local_size
        self.padding = sum([[t - 1 - t // 2, t // 2] for t in local_size[::-1]], [])

    def forward(self, X):
        n, c, h, w = X.shape  # torch.Size([1, 2048, 8, 8])

        # N, C, H, W -> N, C, H + local_size0 - 1, W + local_size1 - 1
        X = F.pad(X, self.padding)

        # N, C, H + local_size0 - 1, W + local_size1 - 1 -> N, C * local_size0 * local_size1, H * W
        X = F.unfold(X, kernel_size=self.local_size)

        # N, C * local_size0 * local_size1, H * W -> N, C, local_size0, local_size1, H * W
        # X = X.view(n, c, *self.local_size, -1)

        # X:  N, C * local_size0 * local_size1, H * W
        return X


class MergeReduce(nn.Module):
    def __init__(self, reduce_method='mean'):
        super(MergeReduce, self).__init__()
        self.reduce_method = reduce_method
        self.local_size = -1

    def register_local_size(self, local_size):
        self.local_size = local_size[0] * local_size[1]
        if self.reduce_method == 'mean':
            self.foo_test = torch.nn.AvgPool2d(local_size, stride=1, padding=local_size[0] // 2, )
        elif self.reduce_method == 'max':
            self.foo_test = torch.nn.MaxPool2d(local_size, stride=1, padding=local_size[0] // 2, )

    def forward(self, X):

        X = X.view(X.shape[0], -1, self.local_size, X.shape[2])
        if self.reduce_method == 'mean':
            return torch.mean(X, dim=2)
        elif self.reduce_method == 'max':
            return torch.max(X, dim=2)

    def forward_test(self, X):
        return self.foo_test(X)


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b * e * f
    return out


class NetE2E(nn.Module):
    def __init__(self, pretrain, net_type, local_size, output_dimension, reduce_function=None, n_noise_points=0, num_stacks=8, num_blocks=1, noise_on_mask=True):
        # output_dimension = 128
        super(NetE2E, self).__init__()
        if net_type == 'vgg_pool4':
            self.net = vgg16('pool4')
        elif net_type == 'vgg_pool5':
            self.net = vgg16('pool5')
        elif net_type == 'resnet50':
            self.net = resnet50(pretrain)
        elif net_type == 'resnet50_pre':
            self.net = resnet50_pre(pretrain)
        elif net_type == 'resnetext':
            self.net = resnetext(pretrain)
        elif net_type == 'resnetupsample':
            self.net = resnetupsample(pretrain)

        self.size_number = local_size[0] * local_size[1]
        self.output_dimension = output_dimension
        # size_number = reduce((lambda x, y: x * y), local_size)
        if reduce_function:
            reduce_function.register_local_size(local_size)
            self.size_number = 1

        self.reduce_function = reduce_function
        self.net_type = net_type
        self.net_stride = net_stride[net_type]
        self.converter = GlobalLocalConverter(local_size)
        self.noise_on_mask = noise_on_mask

        # output_dimension == -1 for abilation study.
        if self.output_dimension == -1:
            self.out_layer = None
        else:
            self.out_layer = nn.Linear(net_out_dimension[net_type] * self.size_number, self.output_dimension) 
            # output_dimension , net_out_dimension[net_type] * size_number
        
        self.n_noise_points = n_noise_points
        # self.norm_layer = lambda x: F.normalize(x, p=2, dim=1)

    # forward
    def forward_test(self, X, return_features=False, do_normalize=True):
        m = self.net.forward(X)

        # not used
        if self.reduce_function:
            X = self.reduce_function.forward_test(m)
        else:
            X = m

        if self.out_layer is None:
            if do_normalize:
                return F.normalize(X, p=2, dim=1)
            else:
                return X
        if self.size_number == 1:
            X = torch.nn.functional.conv2d(X, self.out_layer.weight.unsqueeze(2).unsqueeze(3))
        elif self.size_number > 1:
            X = torch.nn.functional.conv2d(X, self.out_layer.weight.view(self.output_dimension,
                                                                         net_out_dimension[self.net_type],
                                                                         self.size_number).permute(2, 0, 1).reshape(
                self.size_number * self.output_dimension, net_out_dimension[self.net_type]).unsqueeze(2).unsqueeze(3))
        # print('X_new.shape', X.shape)
        # n, c, w, h
        # 1, 128, (w_original - 1) // 32 + 1, (h_original - 1) // 32 + 1
        # X = F.interpolate(X, scale_factor=2, mode='bilinear')
        if do_normalize:
            X = F.normalize(X, p=2, dim=1)

        if return_features:
            return X, m
        else:
            return X

    def forward_step0(self, X, do_normalize=True, return_features=False, **kwargs):
        # downsample_rate = 32
        # pre--X.shape torch.Size([1, 3, 256, 256])
        m = self.net.forward(X)

        if self.out_layer is not None:
            X = torch.nn.functional.conv2d(m, self.out_layer.weight.unsqueeze(2).unsqueeze(3))
        else:
            X = m

        if do_normalize:
            X = F.normalize(X, p=2, dim=1)

        if return_features:
            return X, m
        return X

    def forward_step1(self, X, keypoint_positions, img_shape, obj_mask=None, **kwargs):
        n = X.shape[0]

        # N, C * local_size0 * local_size1, H * W
        X = self.converter(X)

        keypoint_idx = keypoints_to_pixel_index(keypoints=keypoint_positions,
                                                downsample_rate=self.net_stride,
                                                original_img_size=img_shape).type(torch.long)
        # print(keypoint_idx)
        # keypoint_idx tensor([[52, 44, 36, 36, 43, 51, 36, 20, 20, 11, 27, 28, 20, 20, 27, 27]])
        # Never use this reduce_function part.
        if self.reduce_function:
            X = self.reduce_function(X)
        if self.n_noise_points == 0:
            keypoint_all = keypoint_idx
        else:
            if obj_mask is not None:
                obj_mask = F.max_pool2d(obj_mask.unsqueeze(dim=1),
                                        kernel_size=self.net_stride,
                                        stride=self.net_stride,
                                        padding=(self.net_stride - 1) // 2)
                obj_mask = obj_mask.view(obj_mask.shape[0], -1)
                assert obj_mask.shape[1] == X.shape[2], 'mask_: ' + str(obj_mask.shape) + ' fearture_: ' + str(X.shape)
            if self.noise_on_mask:
                keypoint_noise = get_noise_pixel_index(keypoint_idx,
                                                       max_size=X.shape[2],
                                                       n_samples=self.n_noise_points,
                                                       obj_mask=obj_mask)
            else:
                keypoint_noise = get_noise_pixel_index(keypoint_idx,
                                                       max_size=X.shape[2],
                                                       n_samples=self.n_noise_points,
                                                       obj_mask=None)

            keypoint_all = torch.cat((keypoint_idx, keypoint_noise), dim=1)
            # keypoint_all tensor([[51, 43, 27, 35,  0, 51, 35, 19, 19, 11,  0, 27, 19, 19, 27, 35, 13, 17, 28, 33, 47, 45, 46, 21, 20, 42, 62,  8, 15, 10, 18,  5,  2, 16, 49, 34]])

        # n * c * k -> n * k * c
        # N, C * local_size0 * local_size1, H * W - >  #N, H * W, C * local_size0 * local_size1
        X = torch.transpose(X, 1, 2)

        # N, H * W, C * local_size0 * local_size1 -> N, keypoint_all, C * local_size0 * local_size1
        X = batched_index_select(X, dim=1, inds=keypoint_all)
        # torch.stack([X[i, :, keypoint_all[i]] for i in range(n)])

        if self.out_layer is None:
            X = X.view(n, -1, net_out_dimension[self.net_type])
        else:
            X = X.view(n, -1, self.out_layer.weight.shape[0])
        return X

    def forward(self, *args, mode=-1, **kwargs):
        if mode == -1:
            if 'X' in kwargs.keys():
                img_shape = kwargs['X'].shape[2::]
            else:
                img_shape = args[0].shape[2::]
            X = self.forward_step0(*args, **kwargs)
            if len(args):
                args = args[1::]
            if 'X' in kwargs.keys():
                del kwargs['X']
            return self.forward_step1(*args, X=X, img_shape=img_shape, **kwargs)
        elif mode == 0:
            return self.forward_step0(*args, **kwargs)
        elif mode == 1:
            return self.forward_step1(*args, **kwargs)

    def cuda(self, device=None):
        self.net.cuda(device=device)
        if self.out_layer is not None:
            self.out_layer.cuda(device=device)
        return self
