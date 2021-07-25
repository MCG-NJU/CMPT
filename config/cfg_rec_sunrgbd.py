from config.default_config import DefaultConfig
from datetime import datetime
from functools import reduce

class Config_SUNRGBD(DefaultConfig):

    dataset = 'Rec_SUNRGBD'
    starttime = datetime.now().strftime('%b%d_%H-%M-%S')
    seed = 1
    model = 'trecg'  # trecg | trecg_maxpool (do not remove maxpooling operation)
    arch = 'resnet18'
    content_arch = 'resnet18'   # semantic network
    pretrained = 'places'  # imagenet / places
    content_pretrained = 'places'  # imagenet / places

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    workers = 12
    multi_scale_num = 3
    multi_scale_index = [0,1,2]
    lat_layers = [1,2,3]   # from which layers the encoder features are concatenated to the decoder features
    upsample_start_index = 0

    load_size = 256
    if 'alexnet' in arch:
        crop_size = 227
    else:
        crop_size = 224

    num_classes = 19
    resume = False
    norm = 'bn'
    optimizer = 'adam'
    data_root = '/data/dudapeng/datasets/sun_rgbd/data_in_class_mix/conc_data/'
    train_path = data_root + 'train'
    test_path = data_root + 'test'
    lr = 0.0002
    batch_size = 30
    loops_train = 50
    scheduler = 'lambda_linear'
    print_freq = 5
    direction = 'AtoB'
    multi_scale = True   # pyramid translation
    class_weights = ''  # balance/enhance/''
    class_weights_aux = ''
    ms_keys = ['depth']  # data for multiscale sampling
    save = False # save model
    nopool_layers = 1
    fusion = False
    gate = False  # variant C
    gate_layer = 1
    psnr_ssmi = False
    gate_rate = 0.7
    bn_eval = False
    ib_area = 'd'
    dropout = False
    s_net = 'a'  # which variant
    dim_out = 64

    no_trans = False
    if not no_trans:
        loss_types = ['CLS', 'PERCEPTUAL', 'AUX_CLS']  # 'CLS', 'SEMANTIC', 'GAN', 'PIX2PIX', 'AUX_CLS'
    else:
        loss_types = ['CLS']
    block_type = 'residual'

    resume_path = 'xxx.pth'
    sample_model_path = resume_path
    resume_path_rgb = '/home/dudapeng/workspace/pretrained/resnet18_baseline_rgb.pth'
    resume_path_depth = '/home/dudapeng/workspace/pretrained/resnet18_baseline_depth.pth'
    ignore_keys = ['content']
    fix_keys = ['content_model', 'compl_model']
    param_10x_keys = ['up_list','gen_list']
    task_name = ''
    content_layers = 4
    no_lsgan = False
    fix_grad =False

    alpha_cls = 1.0
    alpha_content = 10
    alpha_aux = 1.0
    alpha_gan = 1.0
    alpha_pix2pix = 1.0

    evaluate = True
    use_fake = False  # use generated images to enhance data sampling
    fake_rate = 0.3
    tsne = False
    inference = False
    start_epoch = 0
    vim_cam = False
    vis_layer = 4
    write_loss = True

    if inference:
        task_name = 'inference_' + resume_path.split('/')[-2]




