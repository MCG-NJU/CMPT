from datetime import datetime
class DefaultConfig:
    seed = 1
    lr = 0.0002
    weight_decay = 0.0001
    momentum = 0.9
    writer_path = '/data1/dudapeng/summary/trans2recg/'
    logs = 'logs'
    model = 'resnet18'
    data_root = ''
    resume = False
    resume_path = ''
    accs = []
    accs_RGB = []
    accs_DEPTH = []

    def parse(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def print_args(self):
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, ':', getattr(self, k))

    def keys(self):
        attrs = []
        for p in dir(self):
            if '__' not in p:
                attrs.append(p)
        return attrs

    def __getitem__(self, item):

        return getattr(self, item)

    not_print_keys = [
        'starttime', 'resume_path', 'not_print_keys', 'num_classes',
        'writer_path', 'data_root', 'momentum',
        'task_name', 'sys_args', 'train_path', 'test_path', 'model_path','log_path'
    ]