

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from configs.base_config import base_cfg, Config
from configs.DataPath import TEST_PATH
from model.Models import Build_SiamDCA
from tracker.SiamDCATracker import SiamDCATracker

from evaluate.test import test
from toolkit.datasets import DatasetFactory

import yaml
import argparse
import os

parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--dataset', default='VOT2018', type=str, help='name of dataset')
parser.add_argument('--tracker', default='SiamDCA', type=str, help='name of tracker')
parser.add_argument('--track_cfg', default='VOT2018.yaml', type=str, help='tracking config file')
parser.add_argument('--gpu_id', default='0', type=str, help="gpu id")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def visual_test_single(tracker, track_cfg, dataset):
    """
    choose and test videos in a certain dataset:
    type the video name, print auc and precision score, press 'ESC' to break and type next one
    """
    while True:
        test_video = input('video name: ')
        test(tracker, 'test', track_cfg, dataset, test_video=test_video, save=False, visual=True)


def test_all(tracker, name, track_cfg, dataset):
    test(tracker, name, track_cfg, dataset, test_video='', save=True, visual=False)


def search(tracker, dataset, configs):
    test_num = len(configs)
    for n in range(test_num):
        i, j, k, l, m = configs[n]
        cfg.context_amount = i
        cfg.penalty_k = j
        cfg.window_influence = k
        cfg.size_lr = l
        name = '%.2f-%.2f-%.2f-%.2f' % (i, j, k, l)
        test_all(tracker, name, track_cfg, dataset)


def grid_search(tracker, dataset, context_amount, penalty_k, window_influence, size_lr):
    for i in context_amount:
        for j in penalty_k:
            for k in window_influence:
                for l in size_lr:
                    track_cfg = Config()
                    track_cfg.context_amount = i
                    track_cfg.penalty_k = j
                    track_cfg.window_influence = k
                    track_cfg.size_lr = l
                    name = '%.2f-%.2f-%.2f-%.2f' % (i, j, k, l)
                    test_all(tracker, name, track_cfg, dataset)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    with tf.Session(config=config) as sess:
        set_session(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        with open(os.path.join('experiments', args.tracker, args.track_cfg), 'r', encoding='utf-8') as f:
            cond = f.read()
            cfg = yaml.load(cond)
            f.close()
        base_cfg.merge_from_file(os.path.join('configs', args.tracker, 'base.yaml'))
        model = Build_SiamDCA(choice=base_cfg.MODEL.CHOICE, fpn_in=base_cfg.MODEL.FPN_IN,
                              x_shape=base_cfg.SEARCH_SIZE, z_shape=base_cfg.TEMP_SIZE,
                              num_filters=base_cfg.MODEL.NUM_FILTERS, num_att=base_cfg.MODEL.NUM_ATT,
                              num_att_heads=base_cfg.MODEL.NUM_ATT_HEADS, d_k=base_cfg.MODEL.D_K, d_v=base_cfg.MODEL.D_V,
                              hidden_dims=base_cfg.MODEL.HIDDEN_DIMS, dropout=base_cfg.MODEL.DROPOUT)
        tracker = SiamDCATracker(model, model_cfg=cfg, session=sess)

        track_cfg = Config()
        track_cfg.context_amount = cfg['CONTEXT_AMOUNT']
        track_cfg.penalty_k = cfg['PENALTY_K']
        track_cfg.window_influence = cfg['WINDOW_INFLUENCE']
        track_cfg.size_lr = cfg['SIZE_LR']

        # dataset_ = 'VOT2018'
        dataset_ = args.dataset
        dataset = DatasetFactory.create_dataset(name=dataset_, dataset_root=TEST_PATH[dataset_], load_img=False)

        """
        Test Mode 1: Evaluate the performance of a tracker with corresponding config on the chosen dataset
        """
        test_all(tracker, name=args.tracker + '-new', track_cfg=track_cfg, dataset=dataset)

        """
        Test Mode 2: Evaluate the performance of a tracker with corresponding config on all datasets
        """
        # datasets = list(TEST_PATH.keys())
        # for dataset_ in datasets:
        #     dataset = DatasetFactory.create_dataset(name=dataset_, dataset_root=TEST_PATH[dataset_], load_img=False)
        #     test_all(tracker, name=args.tracker, track_cfg=track_cfg, dataset=dataset)

        """
        Test Mode 3: Visualized evaluation
        """
        # visual_test_single(tracker, track_cfg, dataset)

        """
        Test Mode 4: Hyper Parameters search
        """
        # configs = [[0.50, 0.06, 0.40, 0.35],
        #            [0.50, 0.04, 0.40, 0.40],
        #            [0.50, 0.08, 0.35, 0.35]]
        # search(tracker, dataset, configs)

        """
        Test Mode 5: Grid search
        """
        # context_amount = [0.50]
        # penalty_k = [0.04, 0.06, 0.08]
        # window_influence = [0.35, 0.40]
        # size_lr = [0.35, 0.40]
        # grid_search(tracker, dataset, context_amount, penalty_k, window_influence, size_lr)

