import copy

import numpy as np
import torch
import torch.distributed as dist
import torchvision

import model.utils as util
from . import networks
# from util.average_meter import AverageMeter
from .trans2_model import Trans2Net
from colorama import Fore, Style
import random


class Fusion(Trans2Net):

    def __init__(self, cfg, writer=None, device=None):
        super(Fusion, self).__init__(cfg, writer,device=device)
        # log_keys = ['INTERSECTION_RGB', 'INTERSECTION_DEPTH', 'UNION_RGB', 'UNION_DEPTH', 'LABEL_RGB', 'LABEL_DEPTH']
        # for key in log_keys:
        #     self.loss_meters[key] = util.AverageMeter()
        # self.set_criterion(cfg, self.net_rgb)
        # self.set_criterion(cfg, self.net_depth)

        if 'CLS' in self.cfg.loss_types or self.cfg.EVALUATE:
            criterion_cls = util.CrossEntropyLoss(weight=cfg.class_weights, device=self.device)
            self.net_rgb.set_cls_criterion(criterion_cls)
            self.net_depth.set_cls_criterion(criterion_cls)

        if 'PERCEPTUAL' in self.cfg.loss_types:
            criterion_content = torch.nn.L1Loss()
            content_model = networks.Content_Model(cfg, criterion_content)
            self.net_rgb.set_content_model(content_model)
            self.net_depth.set_content_model(content_model)

        if 'PIX2PIX' in self.cfg.loss_types:
            criterion_pix2pix = torch.nn.L1Loss()
            self.net_rgb.set_pix2pix_criterion(criterion_pix2pix)
            self.net_depth.set_pix2pix_criterion(criterion_pix2pix)

    def _define_networks(self):
        cfg_tmp = copy.deepcopy(self.cfg)
        # cfg_tmp.MODEL = 'trecg'
        # cfg_tmp.loss_types = ['SEMANTIC', 'AUX_CLS']
        # cfg_tmp.MULTI_TARGETS = ['depth_ms']
        # cfg_tmp.RESUME = False
        # cfg_tmp.USE_FAKE_DATA = True
        # cfg_tmp.NO_TRANS = False
        self.net_rgb = networks.define_netowrks(cfg_tmp, device=self.device)
        self.net_depth = networks.define_netowrks(cfg_tmp, device=self.device)
        self.net = networks.Fusion(self.cfg, self.net_rgb, self.net_depth)

    def set_input(self, data):
        input_A = data['image']
        input_B = data['depth']
        self._label = data['label']
        self.input_rgb = input_A.to(self.device)
        self.input_depth = input_B.to(self.device)
        if self.cfg.multi_scale and self.phase == 'train':
            self.input_rgb_ms = [d.to(self.device) for d in data['ms_image']]
            self.input_depth_ms = [d.to(self.device) for d in data['ms_depth']]
        else:
            self.input_rgb_ms = None
            self.input_depth_ms = None

        self.batch_size = self.input_rgb.size(0)
        if 'label' in data.keys():
            self.label = torch.LongTensor(data['label']).to(self.device)
        else:
            self.label = None

    def _forward(self, cal_loss=True):
        self.result = self.net(self.input_rgb, self.input_depth, rgb_ms=self.input_rgb_ms,
                               depth_ms=self.input_depth_ms, label=self.label, phase=self.phase, cal_loss=cal_loss)

    def validate(self, epoch=None):

        self.phase = 'test'

        # switch to evaluate mode
        self.net_rgb.eval()
        self.net_depth.eval()
        # self.evaluator.reset()

        self.pred_index_all_RGB = []
        self.pred_index_all_DEPTH = []
        self.target_index_all = []

        # batch_index = int(self.val_image_num / cfg.BATCH_SIZE)
        # random_id = random.randint(0, batch_index)

        # batch = tqdm(self.val_loader)
        # for data in batch:
        for i, data in enumerate(self.test_loader):
            self.set_input(data)
            with torch.no_grad():
                self._forward(cal_loss=False)

            # self._process_fc()
            self.pred_rgb = self.result['cls_rgb'].data.max(1)[1]
            self.pred_depth = self.result['cls_depth'].data.max(1)[1]
            self._process_fc()

        result = self._cal_mean_acc(self.cfg, self.test_loader)

        for modal in ['RGB', 'DEPTH']:
            mean_acc = result[modal + '_macc']
            accs = getattr(self.cfg, 'accs_' + modal)
            self.loss_meters['VAL_CLS_MEAN_ACC_' + modal].update(mean_acc)
            accs.append(round(mean_acc, 3))
            print('acc history {0}:{1}'.format(modal, ", ".join(str(acc) for acc in accs)))

            if not self.cfg.inference:
                print(Fore.MAGENTA, 'seed: {0}, lr: {1}'.format(self.cfg.seed, self.cfg.current_lr),
                      'mean accuracy {modal} {ite}/{loops}: {val_acc}'.format(
                          modal=modal, ite=epoch, loops=self.cfg.loops_train, val_acc=round(mean_acc * 100, 3)), Style.RESET_ALL)

    def _process_fc(self):

        _, index_rgb = self.result['cls_rgb'].data.topk(1, 1, largest=True)
        _, index_depth = self.result['cls_depth'].data.topk(1, 1, largest=True)

        self.pred_index_all_RGB.extend(list(index_rgb.cpu().numpy()))
        self.pred_index_all_DEPTH.extend(list(index_depth.cpu().numpy()))
        self.target_index_all.extend(list(self._label.numpy()))

    def _cal_mean_acc(self, cfg, data_loader):

        assert len(self.pred_index_all_RGB) == len(data_loader.dataset.imgs)
        assert len(self.pred_index_all_DEPTH) == len(data_loader.dataset.imgs)
        result = dict()
        for modal in ['RGB', 'DEPTH']:
            pred = getattr(self, 'pred_index_all_{}'.format(modal))
            mean_acc, class_accs = util.mean_acc(np.array(self.target_index_all), np.array(pred),
                                     cfg.num_classes,
                                     data_loader.dataset.classes)
            result[modal + '_macc'] = mean_acc
            result[modal + 'class_accs'] = class_accs
        return result

    def write_loss(self, phase, global_step=1):

        loss_types = self.cfg.loss_types

        if phase == 'train':

            self.writer.add_image('source_rgb',
                                  torchvision.utils.make_grid(self.input_rgb.data[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)
            self.writer.add_image('source_depth',
                                  torchvision.utils.make_grid(self.input_depth.data[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)

            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], global_step=global_step)

            if 'CLS' in loss_types:
                self.writer.add_scalar('TRAIN_CLS_LOSS', self.loss_meters['TRAIN_CLS_LOSS'].avg,
                                       global_step=global_step)

            if 'PERCEPTUAL' in loss_types:
                self.writer.add_image('Gen_rgb',
                                      torchvision.utils.make_grid(self.result['gen_rgb'].data[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                self.writer.add_image('Gen_depth',
                                      torchvision.utils.make_grid(self.result['gen_depth'].data[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)

        elif phase == 'test':

            self.writer.add_scalar('VAL_CLS_ACC', self.loss_meters['VAL_CLS_ACC'].val * 100.0,
                                   global_step=global_step)
            self.writer.add_scalar('VAL_CLS_MEAN_ACC', self.loss_meters['VAL_CLS_MEAN_ACC'].val * 100.0,
                                   global_step=global_step)

            if not self.cfg.no_trans:
                self.writer.add_image('/Val_image',
                                      torchvision.utils.make_grid(self.input_rgb.data[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)

                self.writer.add_image('/Val_gen_rgb',
                                      torchvision.utils.make_grid(self.result['gen_rgb'].data.clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                self.writer.add_image('/Val_gen_depth',
                                      torchvision.utils.make_grid(self.result['gen_depth'].data.clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)