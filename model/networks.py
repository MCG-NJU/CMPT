import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet_models
from torch.nn import init

batch_norm = nn.BatchNorm2d
import model.utils as util

def init_weights(net, init_type='normal', key='', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and m.weight is not None and m.weight.requires_grad:
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

def fix_grad(net):
    # print(net.__class__.__name__)

    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('nn.Conv') != -1 or classname.find('BatchNorm2d') != -1:
            m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

    net.apply(fix_func)


def unfix_grad(net):
    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1 or classname.find('Linear') != -1:
            m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True

    net.apply(fix_func)

def define_netowrks(cfg, device=None):

    if cfg.model == 'trecg_maxpool':
        model = TRecgNet_Maxpool(cfg, device=device)
    elif cfg.model == 'trecg':
        model = TrecgNet(cfg, device=device)
    else:
        raise ValueError('model not supported {0}'.format(cfg.model))
    return model


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False)


def conv_norm_relu(dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d,
                   use_leakyRelu=False, use_bias=False, is_Sequential=True):
    if use_leakyRelu:
        act = nn.LeakyReLU(0.2, True)
    else:
        act = nn.ReLU(True)

    if is_Sequential:
        result = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=use_bias),
            norm(dim_out, affine=True),
            act
        )
        return result
    return [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            norm(dim_out, affine=True),
            act]

##############################################################################
# Moduels
##############################################################################
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)

class Base_Model(nn.Module):

    def __init__(self):
        super(Base_Model, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sample_model = None
        self.content_model = None
        self.compl_model = None

        self.hook_handlers = []
        self.hook_feats_in = None
        self.hook_feats_out = None
        self.hook_grads_in = None
        self.hook_grads_out = None

    def get_train_params(self):
        return [p for n, p in self.named_parameters() if not any(item in n for item in self.cfg.fix_keys) and p.requires_grad]

    def get_train_params_1x(self):
        return [p for n, p in self.named_parameters() if not any(item in n for item in self.cfg.param_10x_keys) and not any(item in n for item in self.cfg.fix_keys) and p.requires_grad]

    def get_train_params_10x(self):
        return [p for n, p in self.named_parameters() if any(item in n for item in self.cfg.param_10x_keys) and not any(item in n for item in self.cfg.fix_keys) and p.requires_grad]

    def get_ignore_params(self):
        return [p for n, p in self.named_parameters() if any(item in n for item in self.cfg.fix_keys) and p.requires_grad]

    def save_gradient(self, grad):
        self.gradients = grad

    def bn_eval(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def backprop(self, scores, class_idx):
        loss = self.celoss(scores, class_idx)
        self.zero_grad()
        loss.backward(retain_graph=True)

    def set_sample_model(self, sample_model):
        self.sample_model = sample_model

    def set_content_model(self, content_model):
        self.content_model = content_model
        if self.cfg.s_net == 'b':
            self.fc_aux = copy.deepcopy(self.content_model.fc).requires_grad_()

    def set_compl_model(self, compl_model):
        self.compl_model = compl_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def set_content_criterion(self, criterion):
        self.content_criterion = criterion

    def _hook_layers(self, conv_layer=None, feat=True, grad=True):

        def forward_hook_function(module, ten_in, ten_out):
            self.hook_feats_in = ten_in
            self.hook_feats_out = ten_out

        def backward_hook_function(module, grad_in, grad_out):
            self.hook_grads_in = grad_in[0]
            self.hook_grads_out = grad_out[0]

        layer = 'layer{index}'.format(index=conv_layer)
        if hasattr(self, 'net'):
            model = self.net
        else:
            model = self
        if hasattr(model, layer):
            if feat:
                self.hook_handlers.append(
                    model.__getattr__(layer).register_forward_hook(forward_hook_function)
                )
            if grad:
                self.hook_handlers.append(
                    model.__getattr__(layer).register_backward_hook(backward_hook_function)
                )
    def clear_hooks(self):
        self.hook_feats_in = None
        self.hook_feats_out = None
        self.hook_grads_in = None
        self.hook_grads_out = None
        for h in self.hook_handlers:
            h.remove()

    def ib_act(self, ds_feats, source, act_area=None):

        zeros_t = torch.zeros_like(source)
        ones_t = torch.ones_like(source)
        if act_area == 'similar':
            mask = torch.where(ds_feats > 0, ones_t, zeros_t)
        elif act_area == 'different':
            mask = torch.where(ds_feats > 0, zeros_t, ones_t)
        else:
            raise ValueError('ib area {0} not supported!!!'.format(act_area))

        assert mask.size() == source.size()
        gated = source * mask

        ib_rate = len(torch.nonzero(gated, as_tuple=True)[0]) / len(
           torch.nonzero(source, as_tuple=True)[0])
        if ib_rate == 0 or ib_rate == 1:
            return source

        if gated.ndim == 2:
            return gated

        source_count = torch.where(source > 0, ones_t, zeros_t).sum(dim=(1,2,3))
        gated_count = torch.where(gated > 0, ones_t, zeros_t).sum(dim=(1, 2, 3))
        scale = torch.where(gated_count > 0, source_count / gated_count, torch.ones_like(gated_count))
        gated = gated * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return gated

    def _get_crossdomain_act(self, hook_a, hook_g, gate_strategy='channel'):
        # assert hook_g.ndim in [2, 4]
        assert hook_a.size() == hook_g.size()
        if gate_strategy == 'channel':
            weights = hook_g.detach().mean(axis=(2, 3))
            return F.relu(hook_a.detach() * weights.unsqueeze(-1).unsqueeze(-1))
        elif gate_strategy == 'spatial':
            # spatial
            weights = torch.mean(hook_g.detach(), dim=1, keepdim=True)
            return F.relu(hook_a.detach() * weights)

    def build_upsample_layers(self, block):
        pass

    def forward_encoder(self, source=None, return_layer=None):

        feats = []
        _, _, H, W = source.size()
        x = source
        for l in range(5):
            layer = 'layer{}'.format(str(l))
            x = self.__getattr__(layer)(x)
            feats.append(x)
            if return_layer is not None and l == return_layer:
                return feats
        return feats

    def forward_layerwise(self, x, return_layer=4, mask=None):
        result = {}
        for l in range(return_layer + 1):
            layer = 'layer{index}'.format(index=l)
            x = self.__getattr__(layer)(x)
            if return_layer == 4 and mask is not None and l == self.cfg.gate_layer:
                gated = x * mask
                ib_rate = len(torch.nonzero(gated, as_tuple=True)[0]) / len(
                    torch.nonzero(x, as_tuple=True)[0])
                if 0 < ib_rate <= 1:
                    zeros_t = torch.zeros_like(x)
                    ones_t = torch.ones_like(x)
                    source_count = torch.where(x > 0, ones_t, zeros_t).sum(dim=(1, 2, 3))
                    gated_count = torch.where(gated > 0, ones_t, zeros_t).sum(dim=(1, 2, 3))
                    scale = torch.where(gated_count > 0, source_count / gated_count, torch.ones_like(gated_count))
                    gated = gated * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    result['filter_rate'] = util.cal_filter_rate(x, gated)
                    x = gated
                    result['gated_feats_' + str(l)] = x
            elif self.cfg.dropout and l == self.cfg.gate_layer:
                x = F.dropout2d(x, p=self.cfg.gate_rate)

            result[l] = x
            if l == return_layer:
                return result


class Base_Backbone(nn.Module):

    def __init__(self, cfg, arch):
        super(Base_Backbone, self).__init__()
        self.cfg = cfg
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model = nn.ModuleList([])


    def forward(self, x, layers=None, mask=None):
        result = {}
        for l in range(4 + 1):
            layer = 'layer{index}'.format(index=l)
            x = self.__getattr__(layer)(x)
            if mask is not None and l == self.cfg.gate_layer:
                gated = x * mask
                ib_rate = len(torch.nonzero(gated, as_tuple=True)[0]) / len(
                    torch.nonzero(x, as_tuple=True)[0])
                if 0 < ib_rate < 1:
                    zeros_t = torch.zeros_like(x)
                    ones_t = torch.ones_like(x)
                    source_count = torch.where(x > 0, ones_t, zeros_t).sum(dim=(1, 2, 3))
                    gated_count = torch.where(gated > 0, ones_t, zeros_t).sum(dim=(1, 2, 3))
                    scale = torch.where(gated_count > 0, source_count / gated_count, torch.ones_like(gated_count))
                    gated = gated * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    x = gated

            result[l] = x
        return result


class ResNet_Model(Base_Backbone):

    def __init__(self, cfg, arch, pretrained='imagenet'):
        super(ResNet_Model, self).__init__(cfg, arch)

        if pretrained == 'places':
            resnet_model = resnet_models.__dict__[arch](num_classes=365)
            checkpoint = torch.load('./initmodel/' + arch + '_places365.pth',
                                    map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet_model.load_state_dict(state_dict)
            print('model pretrained using place')
        elif pretrained == 'imagenet':
            resnet_model = resnet_models.__dict__[arch](pretrained=True)
            print('model pretrained using imagenet')
        else:
            resnet_model = resnet_models.__dict__[arch](pretrained=False)

        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool

        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu)
        layer1_nopool = copy.deepcopy(resnet_model.layer1).requires_grad_()
        self.layer1_nopool = layer1_nopool
        self.layer1 = nn.Sequential(self.maxpool, resnet_model.layer1)

        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.fc = resnet_model.fc
        self.avgpool = resnet_model.avgpool

class Content_Model(Base_Model):

    def __init__(self, cfg, criterion=None):
        super(Content_Model, self).__init__()
        self.cfg = cfg
        self.criterion = criterion

        if 'resnet' in cfg.content_arch:
            net = ResNet_Model(cfg, cfg.content_arch, cfg.content_pretrained)

        for l in range(5):
            layer = 'layer{}'.format(l)
            setattr(self, layer, net.__getattr__(layer))


class TRecgNet_Maxpool(Base_Model):

    def __init__(self, cfg, device=None):
        super().__init__()
        self.cfg = cfg
        self.trans = not cfg.no_trans
        self.device = device
        self.arch = cfg.arch

        if 'resnet' in cfg.arch:
            model = ResNet_Model(cfg, cfg.arch, cfg.pretrained)
        else:
            raise ValueError('arch {0} not supported!'.format(cfg.arch))
        self.layer0 = model.layer0
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.maxpool = model.maxpool
        if self.cfg.model == 'trecg':
            nopool_layer = 'layer{}_nopool'.format(cfg.nopool_layers)
            self.__setattr__(nopool_layer, model.__getattr__(nopool_layer))
        self.image_start_index = 1
        self.relu = nn.ReLU()

        self.dims_all = [64, 64, 128, 256, 512]
        self.fc = nn.Linear(self.dims_all[-1], cfg.num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.init_type = 'normal'

        if self.trans:
            if cfg.block_type == 'residual':
                block = Conc_Up_Residual_bottleneck
            elif cfg.block_type == 'simple':
                block = Simple_Upblock

            self.image_start_index = 1
            self.upsample_norm = nn.InstanceNorm2d if self.cfg.norm == 'in' else nn.BatchNorm2d
            self.upsample_times = len(self.dims_all) - 1
            self.build_upsample_layers(block)
            if 'AUX_CLS' in cfg.loss_types:
                if cfg.multi_scale:
                    fc_aux_list = list()
                    for i in range(cfg.multi_scale_num):
                        fc_aux_list.append(nn.Linear(self.dims_all[-i - 1], cfg.num_classes))  # large2small
                    self.fc_aux = nn.ModuleList(fc_aux_list)
                else:
                    self.fc_aux = copy.deepcopy(self.fc).requires_grad_()


        if cfg.pretrained == 'imagenet' or cfg.pretrained == 'places':
            for n, m in self.named_modules():
                if 'sample' in n or 'content' in n:
                    continue
                if 'up' in n or 'fc' in n or 'skip' in n:
                    init_weights(m, self.init_type)
        else:
            init_weights(self, self.init_type)

    def build_upsample_layers(self, block):

        self.gen_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.lat_list = nn.ModuleList()
        dim_in_gen = self.cfg.dim_out

        for i in range(self.upsample_times):

            lat_index = 4 - i - 1
            if lat_index in self.cfg.lat_layers:
                dim_lat = self.dims_all[lat_index]
            else:
                dim_lat = 0

            if i == self.cfg.upsample_start_index:
                dim_in = list(reversed(self.dims_all))[i]
            else:
                dim_in = self.cfg.dim_out

            self.up_list.append(block(dim_in=dim_in, dim_out=self.cfg.dim_out, dim_lat=dim_lat, norm=self.upsample_norm))
            self.gen_list.append(   # size: large to small; channel: few to many
                nn.Sequential(
                    nn.Conv2d(dim_in_gen, 3, 3, 1, 1, bias=False),
                    nn.Tanh()
                )
            )

        if not self.cfg.multi_scale:
            self.gen_list.append(nn.Sequential(
                nn.Conv2d(dim_in_gen, 3, 3, 1, 1, bias=False),
                nn.Tanh()
            ))

        else:
            for i in range(self.cfg.multi_scale_num):
                self.gen_list.append(      # size: large to small; channel: few to many
                    nn.Sequential(
                        nn.Conv2d(self.dims_all[i], self.dims_all[1], 3, 1, 1, bias=False),
                        self.upsample_norm(self.dims_all[1]),
                        self.relu,
                        nn.Conv2d(self.dims_all[1], 3, 3, 1, 1, bias=False),
                        nn.Tanh()
                    )
                )

    def forward_encoder(self, source=None, return_layer=None, mask=None):

        feats = []
        _, _, H, W = source.size()
        x = source
        for l in range(5):
            layer = 'layer{}'.format(str(l))
            x = self.__getattr__(layer)(x)
            feats.append(x)
            if return_layer is not None and l == return_layer:
                return feats
        return feats

    def forward_decoder(self, feats):

        imgs = list()
        feats_re = list(reversed(feats))
        start_index = self.cfg.upsample_start_index
        x = feats_re[start_index]
        for i in range(start_index, self.upsample_times):

            lat_index = 4 - i - 1
            if lat_index in self.cfg.lat_layers:
                lat = feats[lat_index]
            else:
                lat = None
            x = self.up_list[i](x, lat)
            # x = self.up_list[i](x, self.lat_list[i](lat))
            if self.cfg.multi_scale and i >= self.image_start_index:
                gen = F.interpolate(self.gen_list[i](x), scale_factor=2, mode='bilinear', align_corners=True)
                imgs.append(gen)

        if not self.cfg.multi_scale:
            gen = F.interpolate(self.gen_list[-1](x), scale_factor=2, mode='bilinear', align_corners=True)
            imgs.append(gen)
        return imgs

    def forward_fc(self, x):
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x

    def forward_higher(self, x, start_index=None, return_layer=4, cls=False):

        result = {}
        for l in range(start_index, return_layer + 1):
            layer = 'layer{index}'.format(index=l)
            x = self.__getattr__(layer)(x)
            result[l] = x
        if return_layer == 4 and cls:
            cls = self.forward_fc(x)
            result['cls'] = cls
        return result

    def _mix_fake(self, source, target=None, phase='train', sample_model=None, gen_img=None):

        if sample_model is not None:
            with torch.no_grad():
                sample_model.eval()
                result_sample = sample_model(source=target, target=target, label=None,
                                                  phase=phase,
                                                  cal_loss=False)
                fake_imgs = result_sample['gen_img'].detach()
        else:
            fake_imgs = gen_img
        fake_imgs = F.interpolate(fake_imgs, source.size()[2:], mode='bilinear', align_corners=True)
        input_num = len(fake_imgs)
        indexes = [i for i in range(input_num)]
        random_index = random.sample(indexes, int(len(fake_imgs) * self.cfg.fake_rate))

        for i in random_index:
            source[i, :] = fake_imgs.data[i, :]

        return source

    def forward(self, source=None, target=None, target_ms=None, label=None,
                phase='train', loss_types=None, content_layers=None, cal_loss=True, mask_model=None):
        if loss_types is None:
            loss_types = self.cfg.loss_types

        if self.cfg.use_fake and phase == 'train' and self.sample_model is not None:
            source = self._mix_fake(source, target, sample_model=self.sample_model)

        result = {}
        result['source'] = source
        feat = self.forward_encoder(source)
        result['feat'] = feat

        if 'CLS' in self.cfg.loss_types:
            #
            result['cls'] = self.forward_fc(feat[-1])
            if cal_loss:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)
        if self.trans:
            gen_last_feat_list = list()
            target_last_feat_list = list()
            imgs = self.forward_decoder(feat)
            result['gen_img'] = imgs[-1]

            # result_ori = self.content_model(target, cal_loss=False)
            # result['original_feats_4'] = result_ori['feat'][4]

            # if self.cfg.gate:
            if self.cfg.gate and self.cfg.s_net == 'c' and phase == 'train' and random.random() < self.cfg.gate_rate:
                # if mask_model is None:
                #     mask_model = self.content_model
                feats = self.content_model.forward_encoder(target)
                result['original_feats_' + str(self.cfg.vis_layer)] = feats[self.cfg.vis_layer]
                # self.compl_model.eval()
                feats = feats[self.cfg.gate_layer]
                z = feats.detach().requires_grad_()
                result['original_feats_' + str(self.cfg.gate_layer)] = z
                z.register_hook(self.compl_model.save_gradient)
                compl_result = self.compl_model.forward_higher(z, self.cfg.gate_layer + 1, cls=True)
                result['cross_feats_' + str(self.cfg.vis_layer)] = compl_result[self.cfg.vis_layer]
                self.compl_model.backprop(compl_result['cls'], label)
                hook_g = self.compl_model.gradients.clone().detach()
                result['cross_grads'] = hook_g

                if random.random() <= 0.5:
                    guided_act = self._get_crossdomain_act(z, hook_g, gate_strategy='channel')
                else:
                    guided_act = self._get_crossdomain_act(z, hook_g, gate_strategy='spatial')

                # result['ds_feats'] = guided_act
                zeros_t = torch.zeros_like(hook_g)
                ones_t = torch.ones_like(hook_g)
                if self.cfg.ib_area == 'd':
                    mask = torch.where(guided_act > 0, zeros_t, ones_t)
                elif self.cfg.ib_area == 's':
                    mask = torch.where(guided_act > 0, ones_t, zeros_t)
                self.compl_model.gradients = None

            else:
                mask = None

            if 'PERCEPTUAL' in loss_types and cal_loss:

                content_loss = None
                self.content_model.eval()

                if content_layers is None:
                    content_layers = self.cfg.content_layers

                if self.cfg.multi_scale:
                      # 56 112 224
                    gen_feat_list = list()
                    for i, t in enumerate(reversed(target_ms)):  # from small to large
                        if i not in self.cfg.multi_scale_index:
                            continue
                        forward_layer = content_layers - len(target_ms) + 1 + i
                        gen_feats = self.content_model.forward_layerwise(imgs[i], return_layer=forward_layer, mask=mask)
                        gen_feat_list.append(gen_feats[forward_layer])
                        target_feats = self.content_model.forward_layerwise(t, return_layer=forward_layer, mask=mask)

                        if cal_loss:
                            # if i not in self.cfg.multi_scale_index:
                            #     continue
                            alphas = [1 for _ in range(forward_layer + 1)]
                            loss = [alpha * loss for alpha, loss in zip(alphas, [
                                self.content_criterion(gen_feats[i], target_feats[i]) for i in range(forward_layer + 1)])]
                            if content_loss is None:
                                content_loss = sum(loss)
                            else:
                                content_loss += sum(loss)
                            result['loss_content'] = content_loss
                else:
                    gen = F.interpolate(result['gen_img'], source.size()[2:], mode='bilinear', align_corners=True)
                    gen_feats = self.content_model.forward_layerwise(gen, return_layer=content_layers, mask=mask)[content_layers]
                    target_feats = self.content_model.forward_layerwise(target, return_layer=content_layers, mask=mask)[content_layers]

                    if cal_loss:
                        if isinstance(gen_feats, list):
                            alphas = [1 for _ in range(content_layers + 1)]
                            result['loss_content'] = sum([alpha * loss for alpha, loss in zip(alphas, [
                                    self.content_criterion(gen_feats[i], target_feats[i]) for i in range(content_layers + 1)])])
                        else:
                            result['loss_content'] = self.content_criterion(gen_feats, target_feats)

                if mask is not None:
                    result['gated_feats_' + str(self.cfg.vis_layer)] = target_feats[self.cfg.vis_layer]
                else:
                    result['original_feats_' + str(self.cfg.vis_layer)] = target_feats[self.cfg.vis_layer]

                if 'AUX_CLS' in loss_types and cal_loss:
                    loss_aux_cls = None
                    if self.cfg.multi_scale:
                        for i, fc in enumerate(reversed(self.fc_aux)):  # from small to large
                            gen_feat = gen_feat_list[i]
                            pred_aux = fc(nn.AvgPool2d(gen_feat.size()[2], 1)(gen_feat).flatten(1))
                            if loss_aux_cls is None:
                                loss_aux_cls = self.cls_criterion(pred_aux, label)
                            else:
                                loss_aux_cls += self.cls_criterion(pred_aux, label)
                    else:
                        feat_gen = gen_feats[4]
                        pred_aux_gen = self.fc_aux(self.avgpool(feat_gen).flatten(1))
                        loss_aux_cls = self.cls_criterion(pred_aux_gen, label)

                    result['loss_aux_cls'] = loss_aux_cls

            pix2pix_loss = None
            if 'PIX2PIX' in loss_types and cal_loss:
                if self.cfg.multi_scale:
                    for i, t in enumerate(reversed(target_ms)):
                        if pix2pix_loss is None:
                            pix2pix_loss = self.pix2pix_criterion(imgs[i], t)
                        else:
                            pix2pix_loss += self.pix2pix_criterion(imgs[i], t)
                else:
                    pix2pix_loss = self.pix2pix_criterion(result['gen_img'], target)
                result['loss_pix2pix'] = pix2pix_loss

        return result


class TrecgNet(TRecgNet_Maxpool):

    def __init__(self, cfg, device=None):
        super().__init__(cfg, device)
        nopool_layer = 'layer{}'.format(cfg.nopool_layers)
        self.__setattr__(nopool_layer, self.__getattr__(nopool_layer+'_nopool'))
        self.layer1 = self.layer1_nopool
        self.image_start_index = 0
        self.upsample_times = len(self.dims_all) - 2

class Conc_Up_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, dim_lat=0, stride=1, norm=nn.BatchNorm2d, upsample=True, init_type='normal'):
        super(Conc_Up_Residual_bottleneck, self).__init__()

        self.relu = nn.ReLU()
        self.upsample = upsample
        if dim_in == dim_out:
            kernel_size, padding = 3, 1
        else:
            kernel_size, padding = 1, 0

        self.smooth = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            norm(dim_out),
            self.relu,
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out))

        dim_in = dim_out + dim_lat
        dim_med = int(dim_in / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)

        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)
        init_weights(self, init_type=init_type)

    def forward(self, x, y=None):

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.smooth(x)
        residual = x

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        if self.upsample:
            x += residual

        return self.relu(x)


class Simple_Upblock(nn.Module):

    def __init__(self, dim_in, dim_out, dim_lat=0, norm=nn.BatchNorm2d, upsample=True, init_type='normal'):
        super(Simple_Upblock, self).__init__()

        self.relu = nn.ReLU()
        self.upsample = upsample
        self.smooth = conv_norm_relu(dim_in, dim_out, norm=norm)

        dim_in = dim_out + dim_lat
        self.conc = conv_norm_relu(dim_in, dim_out, kernel_size=1, padding=0)
        init_weights(self, init_type=init_type)

    def forward(self, x, y=None):

        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.smooth(x)
        if y is not None:
            x = torch.cat((x, y), 1)
        x = self.conc(x)
        return x

class GANDiscriminator_Image(nn.Module):
    # initializers
    def __init__(self, cfg, device=None):
        super(GANDiscriminator_Image, self).__init__()
        self.cfg = cfg
        self.device = device
        norm = nn.BatchNorm2d
        self.d_downsample_num = 4
        relu = nn.LeakyReLU(0.2)

        distribute = [
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            norm(64),
            relu,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm(128),
            relu,
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm(256),
            relu,
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            norm(256),
            relu,
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            norm(512),
            relu,
            nn.Conv2d(512, 1, kernel_size=1),
        ]

        self.criterion = nn.BCELoss() if cfg.no_lsgan else nn.MSELoss()
        if self.cfg.no_lsgan:
            distribute.append(nn.Sigmoid())

        self.distribute = nn.Sequential(*distribute)
        init_weights(self, 'normal')

    def forward(self, x, target):
        # distribution
        pred = self.distribute(x)

        if target:
            label = 1
        else:
            label = 0

        dis_patch = torch.FloatTensor(pred.size()).fill_(label).to(self.device)
        loss = self.criterion(pred, dis_patch)

        return loss


class Fusion_CDG(Base_Model):

    def __init__(self, cfg, rgb_model=None, depth_model=None, device='cuda'):
        super(Fusion_CDG, self).__init__()
        self.cfg = cfg
        self.device = device
        self.rgb_model = rgb_model
        self.depth_model = depth_model
        self.aux_model = copy.deepcopy(rgb_model)
        self.aux_fc_rgb = copy.deepcopy(rgb_model.fc)
        self.aux_fc_depth = copy.deepcopy(rgb_model.fc)

    def forward(self, input_rgb, input_depth, rgb_ms, depth_ms, label, phase=None, cal_loss=True):
        result = {}

        rgb_result = self.rgb_model(input_rgb, target=input_depth, target_ms=depth_ms, label=label, cal_loss=cal_loss, mask_model=self.depth_model)
        depth_result = self.depth_model(input_depth, target=input_rgb, target_ms=rgb_ms, label=label, cal_loss=cal_loss, mask_model=self.rgb_model)

        if not self.cfg.no_trans:
            result['gen_depth'] = rgb_result['gen_img']
            result['gen_rgb'] = depth_result['gen_img']

        if 'PERCEPTUAL' in self.cfg.loss_types and cal_loss:
            result['loss_content'] = rgb_result['loss_content'] + depth_result['loss_content']
        if 'CLS' in self.cfg.loss_types:
            result['cls_rgb'] = rgb_result['cls']
            result['cls_depth'] = depth_result['cls']
            if cal_loss:
                result['loss_cls'] = rgb_result['loss_cls'] + depth_result['loss_cls']
        if 'AUX_CLS' in self.cfg.loss_types and cal_loss:
            result['loss_aux_cls'] = rgb_result['loss_aux_cls'] + depth_result['loss_aux_cls']

        # if self.cfg.gate and cal_loss:
        #     result['loss_gate'] = rgb_result['loss_gate'] + depth_result['loss_gate']

        return result

class Fusion(Base_Model):

    def __init__(self, cfg, rgb_model=None, depth_model=None, device='cuda'):
        super(Fusion, self).__init__()
        self.cfg = cfg
        self.device = device
        self.rgb_model = rgb_model
        self.depth_model = depth_model

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(self.rgb_model.dims_all[-1] * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, cfg.num_classes)
        )

        if cfg.fix_grad:
            fix_grad(self.rgb_model)
            fix_grad(self.depth_model)

        init_weights(self.fc, 'normal')

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, input_rgb=None, input_depth=None, target_ms=None, label=None,
                phase='train', loss_types=None, content_layers=None, cal_loss=True, mask_model=None):

        result = {}
        rgb_specific = self.rgb_model(input_rgb, target=input_depth, label=label, cal_loss=False)
        depth_specific = self.depth_model(input_depth, target=input_rgb, label=label, cal_loss=False)
        x = torch.cat((rgb_specific['feat'][4], depth_specific['feat'][4]), 1).to(self.device)
        x = self.avgpool(x)
        result['cls'] = self.fc(x)

        if cal_loss:
            if 'CLS' in self.cfg.loss_types:
                result['loss_cls'] = self.cls_criterion(result['cls'], label)
        return result

class AlexNet_Regular(nn.Module):
    def __init__(self):
        super(AlexNet_Regular, self).__init__()
        self.cfg = None
        self.device = None
        self.relu = nn.ReLU(True)
        norm = nn.BatchNorm2d
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm(64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            norm(64),
            self.relu
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm(128),
            self.relu
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            norm(256),
            self.relu,
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            norm(256),
            self.relu,
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            norm(512),
            self.relu,
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            norm(512),
            self.relu,
        )
        self.avgpool = nn.AvgPool2d(7, 1)
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000)
        )
        init_weights(self, 'xavier')

    def forward(self, source, target=None, target_ms=None, label=None, phase='train', cal_loss=True):
        result = dict()
        result['source'] = source
        x = self.relu(self.bn1(self.conv1(source)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        result['cls'] = x
        if cal_loss:
            result['loss_cls'] = self.cls_criterion(x, label)
        return result
