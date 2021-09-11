

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from model.Models import Build_SiamDCA
from tracker.SiamDCATracker import SiamDCATracker
from utils.image import get_frames
from configs.base_config import base_cfg, Config

import yaml
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--gpu_id', default='0', type=str, help="gpu id")
parser.add_argument('--tracker', default='SiamDCA', type=str, help='name of the tracker')
parser.add_argument('--track_cfg', default='VOT2018.yaml', type=str, help='tracking config file')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def demo(tracker, cfg):
    while True:
        video = input('Video Path:')
        video_name = video.split()[0].split('\\')[-1]
        cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
        first_frame = True
        for frame in get_frames(video):
            # Initialization in the first frame
            if first_frame:
                try:
                    # 手动选框
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                    # 固定初始框
                    # init_rect = [336, 165, 26, 61]
                except:
                    exit()
                tracker.init(frame, init_rect, video_name)
                first_frame = False

            # Tracking
            else:
                _, _, k = tracker.track(frame, cfg)
                # Exit if ESC pressed
                if k == 27:
                    break
        cv2.destroyWindow(video_name)


def main(sess):
    base_cfg.merge_from_file(os.path.join('configs', args.tracker, 'base.yaml'))
    model = Build_SiamDCA(choice=base_cfg.MODEL.CHOICE, fpn_in=base_cfg.MODEL.FPN_IN,
                          x_shape=base_cfg.SEARCH_SIZE, z_shape=base_cfg.TEMP_SIZE,
                          num_filters=base_cfg.MODEL.NUM_FILTERS, num_att=base_cfg.MODEL.NUM_ATT,
                          num_att_heads=base_cfg.MODEL.NUM_ATT_HEADS, d_k=base_cfg.MODEL.D_K, d_v=base_cfg.MODEL.D_V,
                          hidden_dims=base_cfg.MODEL.HIDDEN_DIMS, dropout=base_cfg.MODEL.DROPOUT)

    with open(os.path.join('experiments', args.tracker, args.track_cfg), 'r', encoding='utf-8') as f:
        cond = f.read()
        cfg = yaml.load(cond)
        f.close()

    tracker = SiamDCATracker(model, model_cfg=cfg, session=sess)

    track_cfg = Config()
    track_cfg.context_amount = cfg['CONTEXT_AMOUNT']
    track_cfg.penalty_k = cfg['PENALTY_K']
    track_cfg.window_influence = cfg['WINDOW_INFLUENCE']
    track_cfg.size_lr = cfg['SIZE_LR']

    demo(tracker, track_cfg)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    with tf.Session(config=config) as sess:
        set_session(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        main(sess)
