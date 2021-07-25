import numpy as np
from colorama import Fore, Style
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
import os
import cv2
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt
from model.torchcam.cams import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM
import torchvision
from matplotlib import cm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def print_key_params(cfg):
    infos = []
    for key in sorted(set(cfg.keys()) ^ set(cfg.not_print_keys)):
        val = cfg.__getitem__(key)
        if hasattr(val, '__call__'):
            continue
        infos.append('{0}: {1}'.format(key, val))

    print('params: ', Fore.RED + ", ".join(str(i) for i in infos) + Style.RESET_ALL)
    print('sys args: ', cfg.sys_args)

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, device=None):
        super(CrossEntropyLoss, self).__init__()
        if weight:
            weight = torch.FloatTensor(weight).to(device)
            # weight = torch.FloatTensor(weight)

        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)

def print_errors(epoch, cfg, loss_dict, key_words=None):
    losses_avg = []
    for key, loss in loss_dict.items():
        if not any(item in key for item in key_words):
            continue
        if loss.count > 0:
            try:
                losses_avg.append('{0}: {1} '.format(key, round(loss.avg, 5)))
            except Exception as e:
                print(e)
                print('error key:', key)

    losses_avg = ' /// '.join([loss for loss in losses_avg])
    print('Loss avg epoch: {0}/{1}'.format(epoch, cfg.loops_train), Fore.GREEN + losses_avg, Style.RESET_ALL)

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.

    assert (output.dim() in [1, 2, 3])
    # print('output.shape:', output.shape)
    # print('target.shape:', target.shape)
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def mean_acc(target_indice, pred_indice, num_classes, classes=None, file_names=None):
    assert (num_classes == len(classes))

    acc = 0.
    # print('{0} Class Acc Report {1}'.format('#' * 10, '#' * 10))
    class_accs = []
    neg_results = {}
    for i in range(num_classes):
        idx = np.where(target_indice == i)[0]
        index = np.arange(0, len(target_indice[idx]))
        pos_index = index[column_or_1d(target_indice[idx]) == column_or_1d(pred_indice[idx])]
        neg_index = list(set(index).difference(set(pos_index)))
        neg_results[i] = file_names[idx][neg_index]
        class_correct = accuracy_score(target_indice[idx], pred_indice[idx])
        acc += class_correct
        class_accs.append(class_correct)
    print('#' * 30)
    return acc / num_classes, class_accs, neg_results

def column_or_1d(y, warn=False):
    """ Ravel column or 1d numpy array, else raises an error

    Parameters
    ----------
    y : array-like

    warn : boolean, default False
       To control display of warnings.

    Returns
    -------
    y : array

    """
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        return np.ravel(y)

    raise ValueError("bad input shape {0}".format(shape))

def load_checkpoint(net, load_path, ingore_keys = [], load_key='state'):
    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage.cuda())
    weights_to_load = checkpoint[load_key]
    # print('checkpoint_keys: ', weights_to_load.keys())
    print('loading {0} ...'.format(load_path))
    if os.path.isfile(load_path):
        state_dict = net.state_dict()
        load_keys = state_dict.keys()
        # print('model_keys: ', load_keys)
        for k, v in weights_to_load.items():
            k = str.replace(k, 'module.', '')
            if k in load_keys:
                if any(item in k for item in ingore_keys):
                    print('loaded ignore: ', k)
                    continue
                state_dict[k] = v
            else:
                print('not loaded: ', k)
        net.load_state_dict(state_dict)
    else:
        raise ValueError('No checkpoint found at {0}'.format(load_path))

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for name, param in net.named_parameters():
                if 'content_model' in name:
                    param.requires_grad = False
                    continue
                param.requires_grad = requires_grad


def cal_filter_rate(feat_ori, feat_after):
    try:
        ib_rate = len(torch.nonzero(feat_after, as_tuple=True)[0]) / len(
            torch.nonzero(feat_ori, as_tuple=True)[0])

    except ZeroDivisionError:
        raise ValueError('filtered feats are all zeros, can not be divided.')
    return ib_rate

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    alpha = 0.3
    cam = heatmap * (1-alpha) + np.float32(image) * alpha
    return norm_image(cam), (heatmap * 255).astype(np.uint8)

def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


## type: 0 for gradcam; 1 for gradcam++
def vis_feats_with_gradcam(image, feats, grads=None, type=1):
    if grads is not None:
        if type == 0:
            weights = grads.detach().mean(axis=(1, 2))
        elif type == 1:
            grad_2 = grads.pow(2)
            grad_3 = grads.pow(3)
            alpha = grad_2 / (2 * grad_2 + (grad_3 * feats).sum(axis=(1, 2), keepdims=True))
            weights = alpha.squeeze_(0).mul_(torch.relu(grads.squeeze(0))).sum(axis=(1, 2))
        else:
            raise ValueError('cam type not specified.')

        activation_map = F.relu(feats.detach() * weights.unsqueeze(-1).unsqueeze(-1), inplace=True)
    else:
        activation_map = feats

    # heatmap = to_pil_image(activation_map.detach().cpu(), mode='F')
    cam, heatmap = overlay_mask(image, activation_map)
    return cam, heatmap

def _normalize(cams):
    """CAM normalization"""
    cams -= cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1)
    cams /= cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1)

    return cams

def overlay_mask(img, mask, alpha=0.7):

    mask = np.sum(mask.detach().cpu().numpy(), axis=0)
    mask = cv2.resize(mask, (224, 224))
    mask = norm_image(mask)
    # mask = _normalize(mask.sum(dim=0))
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask = np.float32(mask) / 255
    heatmap = mask[..., ::-1]  # gbr to rgb
    overlayed_img = alpha * np.float32(img) + (1 - alpha) * heatmap
    return Image.fromarray(norm_image(overlayed_img)), Image.fromarray(norm_image(heatmap))

def test_cam(model, img_tensor, img_pil, img_np, input_layer='layer0', conv_layer='layer4', class_idx=None, save_path=None, writer=None, ite=1):
    cam_extractors = [GradCAM(model, conv_layer),
                      GradCAMpp(model, conv_layer), SmoothGradCAMpp(model, conv_layer, input_layer),
                      ScoreCAM(model, conv_layer, input_layer)]

    fig, axes = plt.subplots(1, len(cam_extractors) + 1, figsize=(100, 20))
    img_tensor.requires_grad = True
    axes[0].imshow(img_pil)
    axes[0].axis('off')
    axes[0].set_title('input', size=50)

    for idx, extractor in enumerate(cam_extractors):
        idx = idx + 1
        # print(extractor.__class__.__name__)
        model.zero_grad()
        x = model(img_tensor.unsqueeze(0), cal_loss=False)
        scores = x['cls']
        activation_map = extractor(class_idx.item(), scores).cpu()
        extractor.clear_hooks()

        activation_map = activation_map
        # activation_map = _normalize(activation_map.sum(dim=0))
        cam, heatmap = overlay_mask(img_np, activation_map.detach().cpu().numpy(), alpha=0.6)

        axes[idx].imshow(cam)
        axes[idx].axis('off')
        axes[idx].set_title(extractor.__class__.__name__, size=50)

    plt.tight_layout()
    # plt.savefig('checkpoints/test.jpg')
    plt.close()
    if writer is not None:
        writer.add_figure('/Val_CAM', figure=fig, global_step=ite)
    # print()

    # axes[len(cam_extractors) + 1].imshow(B)
    # axes[len(cam_extractors) + 1].axis('off')
    # axes[len(cam_extractors) + 1].set_title('complementary', size=10)

    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    # plt.close()

def tsne_vis(tsne, data, labels, save_dir, epoch, iter, classes):
    low_dim_embs = tsne.fit_transform(data)
    img_path = os.path.join(save_dir, "epoch{0}_iter{1}".format(epoch, iter))
    plot_with_labels(low_dim_embs, labels, img_path, classes)

def plot_with_labels(lowDWeights, labels, save_path, classes):
    plt.cla()
    plt.figure(figsize=(12, 12))
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    plt.scatter(X, Y, c=labels, linewidths=0.5, cmap='tab20')
    # for x, y, s in zip(X, Y, labels):
    #     plt.text(x-0.5, y-0.5, s, fontsize=4)
    # 遍历每个点以及对应标签
    # for x, y, s in zip(X, Y, labels):
    #     c = cm.rainbow(int(255/(len(classes)-1) * s)) # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
    #     plt.text(x, y, s, backgroundcolor=c, fontsize=8)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    # cbar = plt.colorbar(boundaries=np.arange(len(classes) + 1) - 0.5)
    # cbar.set_ticks(np.arange(len(classes)))
    # cbar.set_ticklabels(classes)
    # plt.title('Visualize last layer')
    plt.savefig(save_path)
    plt.close()