import os
import random
import sys

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from config.cfg_rec_sunrgbd import Config_SUNRGBD
from config.cfg_rec_nyud2 import Config_NYUD2
from data.datasets_factory import make_dataset
from model import trans2_model, fusion
import model.utils as utils
import time


unloader_img = torchvision.transforms.ToPILImage()
loader_img = torchvision.transforms.ToTensor()
device = torch.device('cuda')

def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    which_cfg = sys.argv[2]
    if 'sunrgbd' in which_cfg:
        cfg = Config_SUNRGBD()
    elif 'nyu' in which_cfg:
        cfg = Config_NYUD2()
    else:
        raise ValueError('dataset not specified.')
    args = sys.argv[1:]
    print('args:', args)
    cfg.__setattr__('sys_args', ' '.join(args))
    cfg.__setattr__('gpus', list(map(int, sys.argv[1].split(','))))

    for para in args:
        result = para.split('=')
        if len(result) == 1:
            continue
        if len(result) == 2 and result[0] in cfg.keys():
            type_param = type(cfg.__getitem__(result[0]))
            if type_param == bool:
                val = eval(result[1])
            elif type_param == list:
                type_elem = type(cfg.__getitem__(result[0])[0])
                val = [type_elem(i) for i in result[1].split(',')]
            else:
                val = type_param(result[1])
            cfg.__setattr__(result[0], val)
        else:
            raise ValueError('sys args {0} not supported!!!'.format(para))

    if cfg.inference:
        cfg.task_name = 'inference_' + cfg.resume_path.replace(os.sep, '_')
    else:
        task_name = '_'.join(args[1:-1]).replace('=', '_')
        cfg.task_name = task_name

    cfg.__setattr__('model_path',
                    os.path.join('checkpoints', cfg.starttime + '_' + cfg.task_name, str(cfg.seed)))

    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
        print('model_path: {}'.format(cfg.model_path))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    seed = cfg.seed
    print('seed-----------', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_set, test_set = make_dataset(cfg)
    cfg.class_weights = train_set.class_weights

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)

    log_path = os.path.join(cfg.writer_path, cfg.starttime, '_', cfg.task_name + '_seed_' + str(cfg.seed))
    writer = SummaryWriter(logdir=log_path)
    cfg.__setattr__('log_path', log_path)
    # model
    if not cfg.fusion:
        model = trans2_model.Trans2Net(cfg, writer=writer, device=device)
    else:
        model = fusion.Fusion(cfg, writer=writer, device=device)

    model.set_data_loader(train_loader, None, test_loader)
    if cfg.resume:
        utils.load_checkpoint(model.net, cfg.resume_path, ingore_keys=cfg.ignore_keys)
    if cfg.inference:
        start_time = time.time()
        # model = model.to(device)
        print('Inference model...')
        model.evaluate(1)
        print('Inference Time: {0} sec'.format(time.time() - start_time))
    else:
        model.train_parameters(cfg)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()


