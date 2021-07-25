import os
from data.rec_sunrgbd import Rec_SUNRGBD


def make_dataset(cfg):
    if cfg.dataset == 'Rec_SUNRGBD':
        train_set = Rec_SUNRGBD(cfg, split='train')
        test_set = Rec_SUNRGBD(cfg, split='test')
        return train_set, test_set
