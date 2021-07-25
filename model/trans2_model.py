import copy
import os
import time
from collections import defaultdict

import numpy as np
import skimage
import torch
import torch.nn as nn
import torchvision
from colorama import Fore, Style
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import model.utils as util
from . import networks

unloader_img = torchvision.transforms.ToPILImage()
loader_img = torchvision.transforms.ToTensor()

class Trans2Net:

    def __init__(self, cfg, writer=None, device=None):

        self.cfg = cfg
        # self.content_model = None
        self.writer = writer
        self.device = device
        self._define_networks()
        self.params_list = []
        self.set_log_data()
        self.set_criterion(cfg, self.net)

    def _define_networks(self):

        if self.cfg.model != 'fusion':
            self.net = networks.define_netowrks(self.cfg, device=self.device)
        else:
            cfg_tmp = copy.deepcopy(self.cfg)
            cfg_tmp.model = 'trecg'
            self.net_rgb = networks.define_netowrks(cfg_tmp, device=self.device)
            self.net_depth = networks.define_netowrks(cfg_tmp, device=self.device)
            util.load_checkpoint(self.net_rgb, self.cfg.resume_path_rgb)
            util.load_checkpoint(self.net_depth, self.cfg.resume_path_depth)
            self.net = networks.Fusion(self.cfg, self.net_rgb, self.net_depth)

        if self.cfg.use_fake:
            if os.path.isfile(self.cfg.sample_model_path):
                print('Use fake data: sample model is {0}'.format(self.cfg.sample_model_path))
                print('fake ratio:', self.cfg.fake_rate)
                cfg_sample = copy.deepcopy(self.cfg)
                # cfg_sample.model = 'trecg'
                cfg_sample.use_fake = False
                cfg_sample.no_trans = False
                model = networks.define_netowrks(cfg_sample, device=self.device)
                util.load_checkpoint(model, self.cfg.sample_model_path, ingore_keys=['fc'])
                self.net.set_sample_model(model)
            else:
                print('//////No checkpoint found for the sample model, continue to train without the sample model')

        self.net = self.net.to(self.device)
        # self.content_net = self.content_net.to(self.device)

        if 'GAN' in self.cfg.loss_types:
            # self.discriminator = networks.GANDiscriminator(self.cfg, device=self.device)
            self.discriminator = networks.GANDiscriminator_Image(self.cfg, device=self.device)

    def _optimize(self, cal_loss=False, ite=None):

        self._forward(cal_loss, ite=ite)

        if 'GAN' in self.cfg.loss_types:

            util.set_requires_grad(self.net, False)
            util.set_requires_grad(self.discriminator, True)
            fake_d = self.result['gen_img']
            real_d = self.target_modal

            loss_d_fake = self.discriminator(fake_d.detach(), False).mean()
            loss_d_true = self.discriminator(real_d.detach(), True).mean()

            loss_d = (loss_d_fake + loss_d_true) * 0.5
            self.loss_meters['TRAIN_GAN_D_LOSS'].update(loss_d.item(), self.cfg.batch_size)

            self.optimizer_d.zero_grad()
            loss_d.backward()
            self.optimizer_d.step()

        loss_g = self._construct_loss()
        if 'GAN' in self.cfg.loss_types and self.discriminator is not None:
            util.set_requires_grad(self.discriminator, False)
            util.set_requires_grad(self.net, True)

        self.optimizer.zero_grad()
        loss_g.backward()
        self.optimizer.step()

    def set_data_loader(self, train_loader=None, val_loader=None, test_loader=None):

        if train_loader is not None:
            self.train_loader = train_loader
            self.train_image_num = self.train_loader.dataset.__len__()

        if val_loader is not None:
            self.val_loader = val_loader
            self.val_image_num = self.val_loader.dataset.__len__()

        if test_loader is not None:
            self.test_loader = test_loader
            self.test_image_num = self.test_loader.dataset.__len__()

    def set_criterion(self, cfg, net):

        if 'CLS' in self.cfg.loss_types or self.cfg.evaluate:
            criterion_cls = util.CrossEntropyLoss(weight=cfg.class_weights, device=self.device)
            net.set_cls_criterion(criterion_cls)

        if not self.cfg.no_trans and 'PERCEPTUAL' in self.cfg.loss_types:
            criterion_content = torch.nn.L1Loss()
            net.set_content_criterion(criterion_content)

            cfg_content = copy.deepcopy(self.cfg)
            cfg_content.model = 'trecg_maxpool'
            cfg_content.no_trans = True
            content_net = networks.define_netowrks(cfg_content, device=self.device)
            criterion_cls = util.CrossEntropyLoss(weight=cfg.class_weights, device=self.device)
            content_net.set_cls_criterion(criterion_cls)

            if self.cfg.s_net != 'a':
                if self.cfg.direction == 'AtoB':
                    util.load_checkpoint(content_net, self.cfg.resume_path_depth)
                else:
                    util.load_checkpoint(content_net, self.cfg.resume_path_rgb)

                if self.cfg.gate:
                    compl_net = networks.define_netowrks(cfg_content, device=self.device)
                    net.set_compl_model(compl_net)
                    if self.cfg.direction == 'AtoB':
                        util.load_checkpoint(compl_net, self.cfg.resume_path_rgb)
                    else:
                        util.load_checkpoint(compl_net, self.cfg.resume_path_depth)

            networks.fix_grad(content_net)
            net.set_content_model(content_net)

        if 'PIX2PIX' in self.cfg.loss_types:
            criterion_pix2pix = torch.nn.L1Loss()
            net.set_pix2pix_criterion(criterion_pix2pix)

    def set_input(self, data):

        self.data = data
        self.file_names.extend(data['name'])
        self._source = data['image']
        self._label = data['label']
        self.source_modal = self._source.to(self.device)
        if 'label' in data.keys():
            self.label = torch.LongTensor(data['label']).to(self.device)
        else:
            self.label = None

        self.target_modal = data['depth'].to(self.device)
        self.ms_target = None
        if self.phase == 'train' and self.cfg.multi_scale:
            self.ms_target = data['ms_depth']
            self.ms_target = [item.to(self.device) for item in self.ms_target]

    def get_scheduler(self, optimizer):

        if self.cfg.scheduler == 'lambda_exp':

            def lambda_rule(epoch):
                lr_l = (1 - float(epoch) / self.cfg.loops_train) ** 0.9
                lr_l = max(0.2, lr_l)
                return lr_l

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda_rule)

        elif self.cfg.scheduler == 'lambda_linear':

            def lambda_rule(epoch):
                lr_l = 1 - max(0, epoch - self.cfg.loops_train * 0.2) / float(self.cfg.loops_train * 0.8)
                return lr_l

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda_rule)

        elif self.cfg.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.cfg.loops_train * 2 // 3,
                                                        gamma=0.2)
        elif self.cfg.scheduler == 'multi_step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[self.cfg.loops_train // 3,
                                                                         self.cfg.loops_train * 2 // 3],
                                                             gamma=0.5)
        return scheduler


    def train_parameters(self, cfg):

        assert self.cfg.loss_types
        self.optimizers = []
        train_params = [{'params': self.net.get_train_params(), 'lr': cfg.lr}]

        if self.cfg.optimizer == 'sgd':
            # train_params = self.net.parameters()
            # train_params = [{'params': self.net.get_train_params_1x(), 'lr': cfg.lr},
            #                 {'params': self.net.get_train_params_10x(), 'lr': cfg.lr * 5}]
            self.optimizer = torch.optim.SGD(train_params, lr=cfg.lr, momentum=cfg.momentum,
                                             weight_decay=cfg.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(train_params, lr=cfg.lr, betas=(0.5, 0.999))

        self.optimizers.append(self.optimizer)

        if 'GAN' in self.cfg.loss_types:
            self.discriminator = self.discriminator.to(self.device)
            self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_d)

        if len(cfg.gpus) > 1:
            self.net = torch.nn.parallel.DataParallel(self.net)
            if 'GAN' in self.cfg.loss_types:
                self.discriminator = nn.DataParallel(self.discriminator.to(self.device))

        self.schedulers = [self.get_scheduler(optimizer) for optimizer in self.optimizers]

        cfg.__setattr__('accs', [])
        for epoch in range(1, cfg.loops_train + 1):

            print('# Training images num = {0}'.format(self.train_loader.dataset.__len__()))
            start_time = time.time()
            self.phase = 'train'
            self.net.train()

            for key in self.loss_meters:
                self.loss_meters[key].reset()

            # self.flatten_list.clear()
            # self.label_list.clear()
            self.file_names = []
            batch = tqdm(self.train_loader)
            for i, data in enumerate(batch):
                self.set_input(data)
                try:
                    self._optimize(cal_loss=True, ite=i)

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        if self.writer is not None:
                            self.writer.close()
                        raise e

            cfg.__setattr__('current_lr', format(self.optimizers[0].param_groups[0]['lr'], '.3e'))
            print('epoch: {0}/{1}'.format(epoch, cfg.loops_train),
                  Fore.CYAN, 'log_path:{0}'.format(cfg.log_path), Style.RESET_ALL
                  )
            util.print_key_params(cfg)
            util.print_errors(epoch, cfg, self.loss_meters, key_words=['LOSS', 'RATE'])
            print('Training Time: {0} sec'.format(time.time() - start_time))
            if self.cfg.write_loss:
                self.write_loss(phase=self.phase, global_step=epoch)

            if cfg.evaluate and (epoch % self.cfg.print_freq == 0 or epoch > cfg.loops_train - 10):
                self.evaluate(epoch)
                print('Epoch Time with test: {0} sec'.format(time.time() - start_time))

            for scheduler in self.schedulers:
                scheduler.step()

    def _forward(self, cal_loss=True, ite=None):

        if self.cfg.no_trans:
            if self.cfg.model == 'fusion':
                self.result = self.net(input_rgb=self.source_modal, input_depth=self.target_modal, label=self.label, phase=self.phase, cal_loss=cal_loss)
            else:
                self.result = self.net(source=self.source_modal,  label=self.label, phase=self.phase, cal_loss=cal_loss)
        else:
            self.result = self.net(source=self.source_modal, target=self.target_modal, target_ms=self.ms_target, label=self.label,
                                   phase=self.phase, cal_loss=cal_loss)
            self.gen = self.result['gen_img']

    def _construct_loss(self):

        loss_total = torch.zeros(1).to(self.device)
        if 'CLS' in self.cfg.loss_types and 'loss_cls' in self.result:

            cls_loss = self.result['loss_cls'].mean() * self.cfg.alpha_cls
            loss_total += cls_loss

            self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss.item(), self.cfg.batch_size)

        # ) content supervised
        if 'PERCEPTUAL' in self.cfg.loss_types and 'loss_content' in self.result:

            decay_coef = 1
            content_loss = self.result['loss_content'].mean() * self.cfg.alpha_content * decay_coef
            loss_total += content_loss

            self.loss_meters['TRAIN_SEMANTIC_LOSS'].update(content_loss.item(), self.cfg.batch_size)

            if 'loss_gate' in self.result:
                gate_loss = self.result['loss_gate'].mean() * self.cfg.alpha_aux
                loss_total += gate_loss
                self.loss_meters['TRAIN_GATE_LOSS'].update(gate_loss.item(), self.cfg.batch_size)

        if 'AUX_CLS' in self.cfg.loss_types and 'loss_aux_cls' in self.result:

            aux_cls_loss = self.result['loss_aux_cls'].mean() * self.cfg.alpha_aux
            loss_total += aux_cls_loss
            self.loss_meters['TRAIN_AUX_CLS_LOSS'].update(aux_cls_loss.item(), self.cfg.batch_size)

        if 'PIX2PIX' in self.cfg.loss_types and 'loss_pix2pix' in self.result:

            pix2pix_loss = self.result['loss_pix2pix'].mean() * self.cfg.alpha_pix2pix
            loss_total += pix2pix_loss
            self.loss_meters['TRAIN_PIX2PIX_LOSS'].update(pix2pix_loss.item(), self.cfg.batch_size)

        if 'GAN' in self.cfg.loss_types:

            fake_g = self.result['gen_img']
            loss_gan_g = self.discriminator(fake_g, True).mean() * self.cfg.alpha_gan
            self.loss_meters['TRAIN_GAN_G_LOSS'].update(loss_gan_g.item(), self.cfg.batch_size)
            loss_total += loss_gan_g

        return loss_total

    def set_log_data(self):

        self.loss_meters = defaultdict()
        self.log_keys = [
            'TRAIN_GAN_G_LOSS',
            'TRAIN_GAN_D_LOSS',
            'TRAIN_SEMANTIC_LOSS',  # semantic
            'TRAIN_PIX2PIX_LOSS',
            'TRAIN_CONTRAST_LOSS',
            'TRAIN_DISTIL_LOSS',
            'TRAIN_AUX_CLS_LOSS',
            'TRAIN_CLS_ACC',
            'TRAIN_CLS_LOSS',
            'TRAIN_CLS_LOSS_CONTENT',
            'TRAIN_CLS_MEAN_IOU',
            'VAL_CLS_ACC',  # classification
            'VAL_CLS_ACC_RGB',
            'VAL_CLS_ACC_DEPTH',
            'VAL_CLS_LOSS',
            'VAL_CLS_MEAN_IOU',
            'VAL_CLS_MEAN_ACC',
            'VAL_CLS_MEAN_ACC_RGB',
            'VAL_CLS_MEAN_ACC_DEPTH',
            'INTERSECTION',
            'UNION',
            'LABEL',
            'TRAIN_CLS_LOSS_COMPL',
            'TRAIN_CLS_LOSS_FUSE',
            'TRAIN_GATE_LOSS',
            'GEN_FILTER_RATE',
            'TARGET_FILTER_RATE'
        ]
        for item in self.log_keys:
            self.loss_meters[item] = util.AverageMeter()

    def UnNormalize(self, tensor):
        for i in range(len(tensor)):
            tensor[i] = tensor[i] * self.cfg.std[i] + self.cfg.mean[i]
        return tensor

    def tensor2PIL(self, tensor):  # 将tensor-> PIL
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.UnNormalize(image)
        image = ToPILImage()(image)
        return image

    def _cal_psnr_ssmi(self):
        for i in range(len(self.source_modal)):
            target_image = self.tensor2PIL(self.target_modal[i:i + 1]).convert('L')
            gen_image = self.tensor2PIL(self.gen[i:i + 1]).convert('L')
            im1 = np.array(target_image, dtype=np.float32)
            im2 = np.array(gen_image, dtype=np.float32)
            psnr = skimage.measure.compare_psnr(im1, im2, 255)
            ssim = skimage.measure.compare_ssim(im1, im2, data_range=255)
            self.psnr_count += psnr
            self.ssim_count += ssim
            self.count += 1

    def validate(self, epoch=None):

        self.phase = 'test'
        self.net.eval()

        self.pred_index_all = []
        self.target_index_all = []

        self.pred_index_content = []
        self.pred_index_compl = []

        self.count = 0
        self.psnr_count = 0
        self.ssim_count = 0

        psnr_ssmi_flag = False
        self.file_names = []

        batch = tqdm(self.test_loader)

        for i, data in enumerate(batch):
            self.set_input(data)


            with torch.no_grad():
                self._forward(cal_loss=False)
            self._process_fc()

            if not self.cfg.no_trans and self.cfg.psnr_ssmi and self.cfg.loops_train - epoch < 3:
                self._cal_psnr_ssmi()
                psnr_ssmi_flag = True

        if psnr_ssmi_flag:
            print('psnr: {},  ssmi: {}'.format(self.psnr_count/self.count, self.ssim_count/self.count))

        mean_acc, class_accs, neg_results = self._cal_mean_acc(self.cfg, self.test_loader, self.target_index_all, self.pred_index_all)
        for i in range(self.cfg.num_classes):
            np.savetxt(os.path.join(self.cfg.model_path, 'neg_results_{}.txt'.format(i)), neg_results[i], delimiter=',', fmt='%s')
        if not self.cfg.inference:
            print(Fore.MAGENTA, 'seed: {0}, lr: {1}'.format(self.cfg.seed, self.cfg.current_lr),
                  'mean accuracy {ite}/{loops}: {val_acc}'.format(
                      ite=epoch, loops=self.cfg.loops_train, val_acc=round(mean_acc * 100, 3)), Style.RESET_ALL)

        # print('mean_acc_original:', mean_acc)
        self.loss_meters['VAL_CLS_MEAN_ACC'].update(mean_acc)
        self.cfg.accs.append(round(mean_acc, 3))
        print('acc history:{0}'.format(", ".join(str(acc) for acc in self.cfg.accs)))
        np.savetxt(os.path.join(self.cfg.model_path, 'class_acc.txt'), class_accs)

    def _process_fc(self):

        # dist.all_reduce(self.result['cls'])
        self.target_index_all.extend(list(self._label.numpy()))
        _, index = self.result['cls'].data.topk(1, 1, largest=True)
        self.pred_index_all.extend(list(index.cpu().numpy()))

    def _cal_mean_acc(self, cfg, data_loader, target_index_all, pred_index_all):

        assert len(self.pred_index_all) == len(data_loader.dataset.imgs)
        mean_acc, class_accs, neg_results = util.mean_acc(np.array(target_index_all), np.array(pred_index_all),
                                 cfg.num_classes,
                                 data_loader.dataset.classes, np.array(self.file_names))
        return mean_acc, class_accs, neg_results

    def evaluate(self, epoch):
        self.phase = 'test'
        for key in self.loss_meters:
            self.loss_meters[key].reset()
        self.validate(epoch)
        util.print_errors(epoch, self.cfg, self.loss_meters, key_words=['MEAN_ACC'])
        if self.cfg.write_loss:
            self.write_loss(phase=self.phase, global_step=epoch)
        if self.cfg.save and self.cfg.loops_train - epoch <=3:
            outfile = os.path.join(self.cfg.model_path, '{0}.pth'.format(self.cfg.model))
            torch.save({'model': self.cfg.model, 'epoch': epoch, 'seed': self.cfg.seed, 'state': self.net.state_dict()},
                       outfile)

    def inference(self):
        self.phase = 'test'
        start_time = time.time()
        print('Inferencing model...')
        self.evaluate(1)
        print('Inference Time: {0} sec'.format(time.time() - start_time))

    def _normalize(self, cam):

        cam -= np.min(cam)
        cam /= np.max(cam)

        return cam

    def write_loss(self, phase, global_step=1):

        source_modal_show = self.result['source']
        if isinstance(self.target_modal, list):
            target_modal_show = self.target_modal[0]
        else:
            target_modal_show = self.target_modal

        if phase == 'train':

            self.writer.add_scalar('Seed', self.cfg.seed,
                                   global_step=global_step)

            for key, item in self.loss_meters.items():
                if 'TRAIN' in key and item.avg > 0:
                    self.writer.add_scalar(key, item.avg, global_step=global_step)

            self.writer.add_image('/Train_image',
                                  torchvision.utils.make_grid(source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)
            if target_modal_show is not None:
                self.writer.add_image('/Train_target',
                                      torchvision.utils.make_grid(target_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)

            self.writer.add_scalar('/LR', self.optimizer.param_groups[0]['lr'], global_step=global_step)

            if not self.cfg.no_trans:

                self.writer.add_image('/Train_gen',
                                      torchvision.utils.make_grid(self.gen.data[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                self.writer.add_image('/Train_image',
                                      torchvision.utils.make_grid(source_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                # if 'gen_for_content_model' in self.result.keys():
                #     self.writer.add_image('/gen_for_content_model',
                #                           torchvision.utils.make_grid(self.result['gen_for_content_model'].data[:6].clone().cpu().data, 3,
                #                                                       normalize=True), global_step=global_step)
        elif phase == 'test':

            self.writer.add_image('/Val_image',
                                  torchvision.utils.make_grid(source_modal_show.clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)
            if not self.cfg.no_trans:

                self.writer.add_image('/Val_gen',
                                      torchvision.utils.make_grid(self.gen.data.clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
                self.writer.add_image('/Val_target',
                                      torchvision.utils.make_grid(target_modal_show.data.clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)

            self.writer.add_scalar('/VAL_CLS_ACC', self.loss_meters['VAL_CLS_ACC'].val * 100.0,
                                   global_step=global_step)
            self.writer.add_scalar('/VAL_CLS_MEAN_ACC', self.loss_meters['VAL_CLS_MEAN_ACC'].val * 100.0,
                                   global_step=global_step)

