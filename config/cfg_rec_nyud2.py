from config.cfg_rec_sunrgbd import Config_SUNRGBD


class Config_NYUD2(Config_SUNRGBD):

    num_classes = 10
    data_root = '/data/dudapeng/datasets/nyud2/conc_data/'
    train_path = data_root + 'train'
    test_path = data_root + 'test'
