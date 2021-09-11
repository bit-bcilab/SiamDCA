

from yacs.config import CfgNode as CN


class Config(object):
    def __init__(self):
        pass


__C = CN()

base_cfg = __C

__C.META_ARC = "SiamDCA"

# ------------------------------------------------------------------------ #
# Model and Size Settings
# ------------------------------------------------------------------------ #
__C.SEARCH_SIZE = [256, 256, 3]

__C.TEMP_SIZE = [128, 128, 3]

__C.MODEL = CN()

# choice of Backbone
__C.MODEL.CHOICE = 2

# the stages of features
__C.MODEL.FPN_IN = [2, 3, 4]

__C.MODEL.FPN_OUT = [2]

__C.MODEL.STRIDE = 8

__C.MODEL.SCORE_SIZE = [32, 32]

__C.MODEL.NUM_FILTERS = 256

__C.MODEL.NUM_FPN = 2

__C.MODEL.NUM_ATT = 4

__C.MODEL.NUM_ATT_HEADS = 8

__C.MODEL.D_K = 32

__C.MODEL.D_V = 32

__C.MODEL.HIDDEN_DIMS = 2 * __C.MODEL.NUM_FILTERS

__C.MODEL.DROPOUT = 0.1

# ------------------------------------------------------------------------ #
# Train and Data Options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

__C.TRAIN.EPOCHS = 80

__C.TRAIN.BATCH_PER_GPU = 8

# Optimizer settings
__C.TRAIN.INIT_LR = 2.5e-3

__C.TRAIN.MINI_LR = 1e-5

__C.TRAIN.MOMENTUM = 0.90

__C.TRAIN.DECAY = 0.08

__C.TRAIN.WARM_UP = 3

__C.TRAIN.RESTART_EPOCHS = 20

__C.TRAIN.DECAY_EPOCHS = 20

__C.TRAIN.MID_EPOCHS = 18

__C.TRAIN.DECAY_STEPS = 25

__C.TRAIN.MID_LR_RATIO = 1. / 16

__C.TRAIN.UNFREEZE_TIMES = 3

TRAIN_SETTINGS = dict(ENCODE_SETTINGS=dict(pos_num=16, neg_num=42, pos_radium=4., neg_radium=2.),

                      LOSS_WEIGHT=dict(pos=0.5, neg=0.5, iou=0.5, l1=0.025),

                      CROP_SETTINGS=dict(template=dict(context_amount=.5,
                                                       keep_scale_prob=.75, min_scale=.9, max_scale=1.1,
                                                       crop_size_rate=1.,
                                                       keep_center_prob=.75, shift_rate=.1, box_protect_rate=0.),
                                         search=dict(context_amount=.5,
                                                     keep_scale_prob=.25, min_scale=.75, max_scale=1.25,
                                                     crop_size_rate=2.,
                                                     keep_center_prob=.4, shift_rate=.30, box_protect_rate=.2),
                                         val=dict(context_amount=.5,
                                                  keep_scale_prob=1., min_scale=1., max_scale=1., crop_size_rate=2.,
                                                  keep_center_prob=1., shift_rate=.0, box_protect_rate=0.)),

                      AUG_SETTINGS=dict(gray=0.,
                                        mix=0.30,  # 无论正负，图像中没有干扰物体，即只有一个标注框时，在目标附近加入其他干扰物体
                                        translation_other=.15,  # 正样本对时，将目标移动到其他图像上的概率
                                        translation_background=.30,  # 正样本对时，将目标移动到图像上其他位置的概率
                                        neg_threshold=.40,  # 总的负样本对概率
                                        neg_pair=0.08,  # x和z不匹配
                                        occ_background=.24,  # 物体被背景区域完全遮挡概率
                                        occ_object=.40,  # 物体被其他物体完全遮挡概率
                                        template=dict(flip={'threshold': 0.}, rotate={'threshold': .0, 'max_angle': 0.},
                                                      blur={'threshold': 0.},
                                                      motion={'threshold': 0., 'max_degree': 0., 'max_angle': 0.},
                                                      erase={'threshold': 0.},
                                                      pca={'threshold': .0}, color={'threshold': .0}),
                                        search=dict(flip={'threshold': .05},
                                                    rotate={'threshold': .15, 'max_angle': 15.},
                                                    blur={'threshold': .10},
                                                    motion={'threshold': .20, 'max_degree': 8., 'max_angle': 10.},
                                                    erase={'threshold': .30},
                                                    pca={'threshold': .0}, color={'threshold': .10})),

                      DATA_SETTINGS=dict(dataset_used=('DET', 'DET_val', 'COCO', 'COCO_val',
                                                       'VID', 'VID_val', 'LaSOT', 'GOT', 'GOT_val'),
                                         DET=dict(label_path='det-train.json', match_range=1,
                                                  num_data=333474, multiply=1,
                                                  num_val=800, num_val_objects=800,
                                                  num_train=8000, num_train_objects=8000),

                                         DET_val=dict(label_path='det-val.json', match_range=1,
                                                      num_data=18680, multiply=1,
                                                      num_val=200, num_val_objects=200,
                                                      num_train=1500, num_train_objects=1500),

                                         COCO=dict(label_path='coco-train.json', match_range=1,
                                                   num_data=117266, multiply=4,
                                                   num_val=500, num_val_objects=2000,
                                                   num_train=5000, num_train_objects=20000),

                                         COCO_val=dict(label_path='coco-val.json', match_range=1,
                                                       num_data=4952, multiply=4,
                                                       num_val=250, num_val_objects=1000,
                                                       num_train=500, num_train_objects=2000),

                                         VID=dict(label_path='vid-train.json', match_range='all',
                                                  num_data=3862, multiply=4,
                                                  num_val=0, num_val_objects=0,
                                                  num_train=3862, num_train_objects=15000),

                                         VID_val=dict(label_path='vid-val.json', match_range='all',
                                                      num_data=555, multiply=4,
                                                      num_val=0, num_val_objects=0,
                                                      num_train=555, num_train_objects=2000),

                                         LaSOT=dict(label_path='lasot.json', match_range='mix',
                                                    num_data=1400, multiply=10,
                                                    num_val=0, num_val_objects=0,
                                                    num_train=1400, num_train_objects=14000),

                                         GOT=dict(label_path='got-train.json', match_range='mix',
                                                  num_data=9335, multiply=2,
                                                  num_val=0, num_val_objects=0,
                                                  num_train=9335, num_train_objects=18000),

                                         GOT_val=dict(label_path='got-val.json', match_range='mix',
                                                      num_data=180, multiply=2,
                                                      num_val=0, num_val_objects=0,
                                                      num_train=180, num_train_objects=360)))
