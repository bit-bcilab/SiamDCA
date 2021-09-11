

import numpy as np
import tensorflow as tf
import keras.backend as K

from tracker.BaseTracker import BaseSiamTracker, change, sz
from training.Augmentation import random_crop
from tracker.BoxDecoder import ltrb_decoder
from utils.grid import generate_grid
from utils.image import rgb_normalize, get_subwindow
from utils.bbox import corner2center, clip_bbox_center, center2corner, clip_bbox_corner

import random
import cv2


class SiamDCATracker(BaseSiamTracker):
    def __init__(self, model, model_cfg, session):
        super(BaseSiamTracker, self).__init__()
        self.model = model
        self.mode = model_cfg['MODE']

        self.crop_settings_temp = model_cfg['CROP_SETTINGS_TEMP']
        self.crop_size_rate_z = self.crop_settings_temp['crop_size_rate']
        self.crop_settings_search = model_cfg['CROP_SETTINGS_SEARCH']
        self.search_size = model_cfg['SEARCH_SIZE'][1]
        self.template_size = model_cfg['TEMPLATE_SIZE'][1]

        self.grid = generate_grid(model_cfg['SEARCH_SIZE'], model_cfg['SCORE_SIZE'])
        hanning = np.hanning(model_cfg['SCORE_SIZE'][0])
        window_ = np.outer(hanning, hanning)
        self.window = window_.astype(np.float32)

        num_filters, nz = model.layers[-2].outputs[0].shape.as_list()[1:3]
        self.x = tf.placeholder(tf.float32, shape=[None] + model_cfg['SEARCH_SIZE'])
        self.z = tf.placeholder(tf.float32, shape=[None] + model_cfg['TEMPLATE_SIZE'])
        self.zf_t = [tf.placeholder(tf.float32, shape=[None, num_filters, nz]) for i in range(3)] + \
                    [tf.placeholder(tf.float32, shape=[None, nz, num_filters]) for i in range(3)]

        self.score, self.bbox = self.build_graph()

        self.model.load_weights(model_cfg['WEIGHT_PATH'], by_name=True)
        self.sess = session

    def build_graph(self):
        zf = self.model.layers[2](self.z)
        zf = self.model.layers[3](zf)
        self.zf = self.model.layers[4](zf)

        xf = self.model.layers[2](self.x)
        xf = self.model.layers[3](xf)
        cls, loc = self.model.layers[5](xf + self.zf_t)
        return ltrb_decoder(cls, loc, self.grid)

    def init(self, img, bbox, video_name=None):
        self.video_name = video_name
        self.channel_average = np.mean(img, axis=(0, 1))
        self.image_shape = img.shape[:2]
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])
        self.target = img[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(bbox[0] + bbox[2]), :]

        if self.mode == 'NFS':
            self.crop_settings_temp['crop_size_rate'] = fix_temp_area(bbox) * self.crop_size_rate_z

        box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)
        z_crop, z_box = random_crop(img, box, self.template_size, self.crop_settings_temp)
        self.z_crop = rgb_normalize(z_crop, mobilenet=False)

        self.zf_v = self.sess.run(self.zf, feed_dict={self.z: self.z_crop, K.learning_phase(): 0})

        self.s_x0 = round(np.sqrt(np.prod(self.size + self.crop_settings_search['context_amount'] * np.sum(self.size)))
                          * self.crop_settings_search['crop_size_rate'])
        self.lost_num = 0
        self.success = 1

        if self.mode == 'NFS':
            self.ratios = [0.76, 0.85, 0.94, 1., 1.08, 1.16]
            zf_v = []
            for i in range(len(self.zf_v)):
                zf_v_ = self.zf_v[i].copy()
                zf_v_ = np.tile(zf_v_, (len(self.ratios), 1, 1))
                zf_v.append(zf_v_)
            self.zf_v_ = zf_v

    def track(self, img, track_cfg):
        # UAV数据集大多为长时序列，出视野干扰比较严重
        # 采用置信度判断丢失+扩大检测范围+得分超过阈值的强寻回策略
        # 因为长时序列中目标容易发生剧烈变化，因此不对搜索区域的尺寸进行限制
        if 'UAV' in self.mode:
            return self.track_uav(img, track_cfg)

        # OTB数据集大多为简单的短序列，少数序列会因为短暂遮挡而无法跟踪
        # 采用松散的丢失判断与寻回策略，但对搜索区域的尺寸变化采取强抑制
        elif 'OTB' in self.mode:
            return self.track_otb(img, track_cfg)

        elif 'LT' in self.mode:
            return self.track_lt(img, track_cfg)

        # 在测试VOT数据集或评测其他短时序列时，只对搜索区域范围变化进行较弱的抑制
        # 避免搜索区域过大时带着大量背景一起跟踪/过小时只能跟踪目标外观上的一小块区域，而无法恢复到正常的搜索范围
        elif 'VOT' in self.mode:
            return self.track_vot(img, track_cfg)

        # 采用SiamFC的多尺度搜索策略，设置多个从大到小的搜索区域
        elif 'NFS' in self.mode:
            return self.track_nfs(img, track_cfg)

        # 从KCF到SiamRPN++一路沿袭至今的后处理pipeline
        else:
            return self.track_normal(img, track_cfg)

    def track_normal(self, img, track_cfg):
        context_amount = track_cfg.context_amount
        penalty_k = track_cfg.penalty_k
        window_influence = track_cfg.window_influence
        lr = track_cfg.size_lr

        # 确定搜索范围的大小，并计算如何将搜索patch坐标下相对位置box转化到整幅图像的绝对位置box
        s_x = round(np.sqrt(np.prod(self.size + context_amount * np.sum(self.size))) *
                    self.crop_settings_search['crop_size_rate'])
        scales = s_x / self.search_size
        offset = np.floor(self.center_pos - s_x / 2.)
        offset = np.concatenate([offset, offset])

        # 将搜索区域图像裁剪出来，并进行与训练时相同的数据预处理操作（rgb通道顺序转换、归一化等）
        window = get_subwindow(img, self.center_pos, self.search_size, s_x, self.channel_average)
        x_crop = rgb_normalize(window, mobilenet=False)

        # Inference
        score, pred_lrtb = self.sess.run([self.score, self.bbox], feed_dict={self.x: x_crop,
                                                                             self.zf_t[0]: self.zf_v[0],
                                                                             self.zf_t[1]: self.zf_v[1],
                                                                             self.zf_t[2]: self.zf_v[2],
                                                                             self.zf_t[3]: self.zf_v[3],
                                                                             self.zf_t[4]: self.zf_v[4],
                                                                             self.zf_t[5]: self.zf_v[5],
                                                                             K.learning_phase(): 0})

        # 将搜索patch坐标下相对位置box转化到整幅图像的绝对位置box，并防止box超出图像范围
        score = score[0]
        pred_lrtb = pred_lrtb[0]
        pred_corner = pred_lrtb * scales  # 在搜索区域上的位置
        pred_corner = pred_corner + offset  # 在原图像上的位置
        pred_xywh = corner2center(pred_corner)
        pred_xywh = clip_bbox_center(pred_xywh, self.image_shape)

        # 先加尺寸抑制再加余弦窗抑制
        s_c = change(sz(pred_xywh[..., 2], pred_xywh[..., 3], context_amount=context_amount) /
                     (sz(self.size[0], self.size[1], context_amount=context_amount)))
        r_c = change((self.size[0] / self.size[1]) / (pred_xywh[..., 2] / pred_xywh[..., 3]))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score
        wscore = (1. - window_influence) * pscore + window_influence * self.window

        # 最高得分处即为目标中心所在位置
        wscore_ = np.reshape(wscore, (-1,))
        index = np.argmax(wscore_)
        pred_xywh_ = np.reshape(pred_xywh, (-1, 4))
        bbox_xywh = pred_xywh_[index, :]

        # 对box尺寸进行滑动平均
        bbox = np.concatenate([bbox_xywh[:2] - bbox_xywh[2:] / 2., bbox_xywh[:2] + bbox_xywh[2:] / 2.])
        bbox_xywh[2:] = self.size * (1 - lr) + bbox_xywh[2:] * lr
        self.center_pos = bbox_xywh[:2]
        self.size = bbox_xywh[2:]

        # demo模式下先显示，再传回box
        if self.video_name is not None:
            box = list(map(int, bbox))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow(self.video_name, img)
            k = cv2.waitKey(30) & 0xff
            return bbox, bbox_xywh, k
        # 其他不可视的模式下，返回最高得分与rect形式的box
        else:
            score_ = score.reshape((-1,))
            outputs = {'bbox': [bbox[0], bbox[1], bbox_xywh[2], bbox_xywh[3]],
                       'best_score': score_[index]}
            return outputs

    def track_vot(self, img, track_cfg):
        context_amount = track_cfg.context_amount
        penalty_k = track_cfg.penalty_k
        window_influence = track_cfg.window_influence
        lr = track_cfg.size_lr

        s_x = round(np.sqrt(np.prod(self.size + context_amount * np.sum(self.size))) *
                    self.crop_settings_search['crop_size_rate'])
        if s_x < self.s_x0 * 0.40:
            s_x = s_x * 1.15
        if s_x > self.s_x0 * 3.5:
            s_x = s_x * 0.90

        scales = s_x / self.search_size
        offset = np.floor(self.center_pos - s_x / 2.)
        offset = np.concatenate([offset, offset])

        window = get_subwindow(img, self.center_pos, self.search_size, s_x, self.channel_average)
        x_crop = rgb_normalize(window, mobilenet=False)

        score, pred_lrtb = self.sess.run([self.score, self.bbox], feed_dict={self.x: x_crop,
                                                                             self.zf_t[0]: self.zf_v[0],
                                                                             self.zf_t[1]: self.zf_v[1],
                                                                             self.zf_t[2]: self.zf_v[2],
                                                                             self.zf_t[3]: self.zf_v[3],
                                                                             self.zf_t[4]: self.zf_v[4],
                                                                             self.zf_t[5]: self.zf_v[5],
                                                                             K.learning_phase(): 0})
        score = score[0]
        pred_lrtb = pred_lrtb[0]
        pred_corner = pred_lrtb * scales
        pred_corner = pred_corner + offset
        pred_xywh = corner2center(pred_corner)
        pred_xywh = clip_bbox_center(pred_xywh, self.image_shape)

        s_c = change(sz(pred_xywh[..., 2], pred_xywh[..., 3], context_amount=context_amount) /
                     (sz(self.size[0], self.size[1], context_amount=context_amount)))
        r_c = change((self.size[0] / self.size[1]) / (pred_xywh[..., 2] / pred_xywh[..., 3]))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score
        wscore = (1. - window_influence) * pscore + window_influence * self.window

        wscore_ = np.reshape(wscore, (-1,))
        index = np.argmax(wscore_)
        pred_xywh_ = np.reshape(pred_xywh, (-1, 4))
        bbox_xywh = pred_xywh_[index, :]

        bbox = np.concatenate([bbox_xywh[:2] - bbox_xywh[2:] / 2., bbox_xywh[:2] + bbox_xywh[2:] / 2.])
        bbox_xywh[2:] = self.size * (1 - lr) + bbox_xywh[2:] * lr
        self.center_pos = bbox_xywh[:2]
        self.size = bbox_xywh[2:]

        if self.video_name is not None:
            box = list(map(int, bbox))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow(self.video_name, img)
            k = cv2.waitKey(30) & 0xff
            return bbox, bbox_xywh, k
        else:
            score_ = score.reshape((-1,))
            outputs = {'bbox': [bbox[0], bbox[1], bbox_xywh[2], bbox_xywh[3]],
                       'best_score': score_[index]}
            return outputs

    def track_otb(self, img, track_cfg):
        context_amount = track_cfg.context_amount
        penalty_k = track_cfg.penalty_k
        window_influence = track_cfg.window_influence
        lr = track_cfg.size_lr

        s_x = round(np.sqrt(np.prod(self.size + context_amount * np.sum(self.size))) *
                    self.crop_settings_search['crop_size_rate'])
        if s_x < self.s_x0 * 0.40:
            if s_x < self.s_x0 * 0.25:
                s_x = self.s_x0 * 0.35
            else:
                s_x = s_x * 1.15
        if s_x > self.s_x0 * 3.0:
            if s_x > self.s_x0 * 5.0:
                s_x = s_x * 0.75
            else:
                s_x = s_x * 0.90

        scales = s_x / self.search_size
        offset = np.floor(self.center_pos - s_x / 2.)
        offset = np.concatenate([offset, offset])

        # 将搜索区域图像裁剪出来，并进行与训练相同的数据预处理操作（rgb通道顺序转换、归一化等）
        window = get_subwindow(img, self.center_pos, self.search_size, s_x, self.channel_average)
        # window, _ = random_crop_local(img, self.box, self.search_size, self.crop_settings_search)
        x_crop = rgb_normalize(window, mobilenet=False)

        score, pred_lrtb = self.sess.run([self.score, self.bbox], feed_dict={self.x: x_crop,
                                                                             self.zf_t[0]: self.zf_v[0],
                                                                             self.zf_t[1]: self.zf_v[1],
                                                                             self.zf_t[2]: self.zf_v[2],
                                                                             self.zf_t[3]: self.zf_v[3],
                                                                             self.zf_t[4]: self.zf_v[4],
                                                                             self.zf_t[5]: self.zf_v[5],
                                                                             K.learning_phase(): 0})
        score = score[0]
        pred_lrtb = pred_lrtb[0]
        pred_corner = pred_lrtb * scales  # 在搜索区域上的位置
        pred_corner = pred_corner + offset  # 在原图像上的位置
        pred_xywh = corner2center(pred_corner)
        pred_xywh = clip_bbox_center(pred_xywh, self.image_shape)

        s_c = change(sz(pred_xywh[..., 2], pred_xywh[..., 3], context_amount=context_amount) /
                     (sz(self.size[0], self.size[1], context_amount=context_amount)))
        r_c = change((self.size[0] / self.size[1]) / (pred_xywh[..., 2] / pred_xywh[..., 3]))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score
        wscore = (1. - window_influence) * pscore + window_influence * self.window

        wscore_ = np.reshape(wscore, (-1,))
        index = np.argmax(wscore_)
        pred_xywh_ = np.reshape(pred_xywh, (-1, 4))
        bbox_xywh = pred_xywh_[index, :]

        score_ = np.reshape(score, (-1,))
        max_score = np.max(score_)
        bbox = np.concatenate([bbox_xywh[:2] - bbox_xywh[2:] / 2., bbox_xywh[:2] + bbox_xywh[2:] / 2.])
        bbox_xywh[2:] = self.size * (1 - lr) + bbox_xywh[2:] * lr
        self.center_pos = bbox_xywh[:2]
        self.size = bbox_xywh[2:]

        if self.success >= 0:
            if max_score < 0.6:
                self.success -= 1
                self.center_pos_lost = self.center_pos
                self.size_lost = self.size
                self.s_x_lost = s_x
                self.lost_frame = 0
            else:
                self.success = 1

        else:
            if self.lost_frame > 3:
                search_patches, offset_, scale_, num = build_detect_area(img, self.lost_frame, self.center_pos_lost,
                                                                         self.s_x_lost, self.search_size, max_num=64)
                zf_v = []
                for i in range(len(self.zf_v)):
                    zf_v_ = self.zf_v[i].copy()
                    zf_v_ = np.tile(zf_v_, (num, 1, 1))
                    zf_v.append(zf_v_)
                score_, pred_lrtb_ = self.sess.run([self.score, self.bbox], feed_dict={self.x: search_patches,
                                                                                       self.zf_t[0]: zf_v[0],
                                                                                       self.zf_t[1]: zf_v[1],
                                                                                       self.zf_t[2]: zf_v[2],
                                                                                       self.zf_t[3]: zf_v[3],
                                                                                       self.zf_t[4]: zf_v[4],
                                                                                       self.zf_t[5]: zf_v[5],
                                                                                       K.learning_phase(): 0})
                pred_corner_ = pred_lrtb_ * scale_  # 在搜索区域上的位置
                pred_corner_ = pred_corner_ + offset_  # 在原图像上的位置
                pred_xywh_ = corner2center(pred_corner_)
                max_ = score_.max(axis=(1, 2))
                if max_.max() > 0.95:
                    index = max_.argmax()
                    max_search_score = score_[index]
                    max_search_score = np.reshape(max_search_score, (-1,))
                    max_pred_xywh_ = np.reshape(pred_xywh_[index], (-1, 4))
                    index = np.argmax(max_search_score)
                    bbox_xywh = max_pred_xywh_[index, :]

                    bbox = np.concatenate([bbox_xywh[:2] - bbox_xywh[2:] / 2., bbox_xywh[:2] + bbox_xywh[2:] / 2.])
                    self.center_pos = bbox_xywh[:2]
                    self.size = bbox_xywh[2:]
                    self.success = 1
                    self.lost_frame = 0
                else:
                    self.lost_frame += 1
            else:
                self.lost_frame += 1

        if self.video_name is not None:
            box = list(map(int, bbox))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow(self.video_name, img)
            k = cv2.waitKey(15) & 0xff
            return bbox, bbox_xywh, k
        else:
            score_ = score.reshape((-1,))
            outputs = {'bbox': [bbox[0], bbox[1], bbox_xywh[2], bbox_xywh[3]],
                       'best_score': score_[index]}
            return outputs

    def track_uav(self, img, track_cfg):
        context_amount = track_cfg.context_amount
        penalty_k = track_cfg.penalty_k
        window_influence = track_cfg.window_influence
        lr = track_cfg.size_lr

        s_x = round(np.sqrt(np.prod(self.size + context_amount * np.sum(self.size))) *
                    self.crop_settings_search['crop_size_rate'])

        scales = s_x / self.search_size
        offset = np.floor(self.center_pos - s_x / 2.)
        offset = np.concatenate([offset, offset])

        window = get_subwindow(img, self.center_pos, self.search_size, s_x, self.channel_average)
        x_crop = rgb_normalize(window, mobilenet=False)

        score, pred_lrtb = self.sess.run([self.score, self.bbox], feed_dict={self.x: x_crop,
                                                                             self.zf_t[0]: self.zf_v[0],
                                                                             self.zf_t[1]: self.zf_v[1],
                                                                             self.zf_t[2]: self.zf_v[2],
                                                                             self.zf_t[3]: self.zf_v[3],
                                                                             self.zf_t[4]: self.zf_v[4],
                                                                             self.zf_t[5]: self.zf_v[5],
                                                                             K.learning_phase(): 0})
        score = score[0]
        pred_lrtb = pred_lrtb[0]
        pred_corner = pred_lrtb * scales
        pred_corner = pred_corner + offset
        pred_xywh = corner2center(pred_corner)
        pred_xywh = clip_bbox_center(pred_xywh, self.image_shape)

        s_c = change(sz(pred_xywh[..., 2], pred_xywh[..., 3], context_amount=context_amount) /
                     (sz(self.size[0], self.size[1], context_amount=context_amount)))
        r_c = change((self.size[0] / self.size[1]) / (pred_xywh[..., 2] / pred_xywh[..., 3]))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score
        wscore = (1. - window_influence) * pscore + window_influence * self.window

        wscore_ = np.reshape(wscore, (-1,))
        index = np.argmax(wscore_)
        pred_xywh_ = np.reshape(pred_xywh, (-1, 4))
        bbox_xywh = pred_xywh_[index, :]

        score_ = np.reshape(score, (-1,))
        max_score = np.max(score_)
        bbox = np.concatenate([bbox_xywh[:2] - bbox_xywh[2:] / 2., bbox_xywh[:2] + bbox_xywh[2:] / 2.])
        bbox_xywh[2:] = self.size * (1 - lr) + bbox_xywh[2:] * lr
        self.center_pos = bbox_xywh[:2]
        self.size = bbox_xywh[2:]

        if self.success >= 0:
            if max_score < 0.6:
                self.success -= 1
                self.center_pos_lost = self.center_pos
                self.size_lost = self.size
                self.s_x_lost = s_x
                self.lost_frame = 0
            else:
                self.success = 1

        else:
            search_patches, offset_, scale_, num = build_detect_area(img, self.lost_frame, self.center_pos_lost,
                                                                     self.s_x_lost, self.search_size, max_num=64)
            zf_v = []
            for i in range(len(self.zf_v)):
                zf_v_ = self.zf_v[i].copy()
                zf_v_ = np.tile(zf_v_, (num, 1, 1))
                zf_v.append(zf_v_)
            score_, pred_lrtb_ = self.sess.run([self.score, self.bbox], feed_dict={self.x: search_patches,
                                                                                   self.zf_t[0]: zf_v[0],
                                                                                   self.zf_t[1]: zf_v[1],
                                                                                   self.zf_t[2]: zf_v[2],
                                                                                   self.zf_t[3]: zf_v[3],
                                                                                   self.zf_t[4]: zf_v[4],
                                                                                   self.zf_t[5]: zf_v[5],
                                                                                   K.learning_phase(): 0})
            pred_corner_ = pred_lrtb_ * scale_
            pred_corner_ = pred_corner_ + offset_
            pred_xywh_ = corner2center(pred_corner_)
            max_ = score_.max(axis=(1, 2))
            if max_.max() > 0.95:
                index = max_.argmax()
                max_search_score = score_[index]
                max_search_score = np.reshape(max_search_score, (-1,))
                max_pred_xywh_ = np.reshape(pred_xywh_[index], (-1, 4))
                index = np.argmax(max_search_score)
                bbox_xywh = max_pred_xywh_[index, :]

                bbox = np.concatenate([bbox_xywh[:2] - bbox_xywh[2:] / 2., bbox_xywh[:2] + bbox_xywh[2:] / 2.])
                self.center_pos = bbox_xywh[:2]
                self.size = bbox_xywh[2:]
                self.success = 1
                self.lost_frame = 0
            else:
                self.lost_frame += 1

        if self.video_name is not None:
            box = list(map(int, bbox))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow(self.video_name, img)
            k = cv2.waitKey(15) & 0xff
            return bbox, bbox_xywh, k
        else:
            score_ = score.reshape((-1,))
            outputs = {'bbox': [bbox[0], bbox[1], bbox_xywh[2], bbox_xywh[3]],
                       'best_score': score_[index]}
            return outputs

    def track_lt(self, img, track_cfg):
        context_amount = track_cfg.context_amount
        penalty_k = track_cfg.penalty_k
        window_influence = track_cfg.window_influence
        lr = track_cfg.size_lr

        s_x = round(np.sqrt(np.prod(self.size + context_amount * np.sum(self.size))) *
                    self.crop_settings_search['crop_size_rate'])

        if s_x < self.s_x0 * 0.35:
            if s_x < self.s_x0 * 0.25:
                s_x = self.s_x0 * 0.30
            else:
                s_x = s_x * 1.15
        if s_x > self.s_x0 * 3.5:
            s_x = s_x * 0.90

        scales = s_x / self.search_size
        offset = np.floor(self.center_pos - s_x / 2.)
        offset = np.concatenate([offset, offset])

        window = get_subwindow(img, self.center_pos, self.search_size, s_x, self.channel_average)
        x_crop = rgb_normalize(window, mobilenet=False)

        score, pred_lrtb = self.sess.run([self.score, self.bbox], feed_dict={self.x: x_crop,
                                                                             self.zf_t[0]: self.zf_v[0],
                                                                             self.zf_t[1]: self.zf_v[1],
                                                                             self.zf_t[2]: self.zf_v[2],
                                                                             self.zf_t[3]: self.zf_v[3],
                                                                             self.zf_t[4]: self.zf_v[4],
                                                                             self.zf_t[5]: self.zf_v[5],
                                                                             K.learning_phase(): 0})
        score = score[0]
        pred_lrtb = pred_lrtb[0]
        pred_corner = pred_lrtb * scales
        pred_corner = pred_corner + offset
        pred_xywh = corner2center(pred_corner)
        pred_xywh = clip_bbox_center(pred_xywh, self.image_shape)

        s_c = change(sz(pred_xywh[..., 2], pred_xywh[..., 3], context_amount=context_amount) /
                     (sz(self.size[0], self.size[1], context_amount=context_amount)))
        r_c = change((self.size[0] / self.size[1]) / (pred_xywh[..., 2] / pred_xywh[..., 3]))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score
        wscore = (1. - window_influence) * pscore + window_influence * self.window

        wscore_ = np.reshape(wscore, (-1,))
        index = np.argmax(wscore_)
        pred_xywh_ = np.reshape(pred_xywh, (-1, 4))
        bbox_xywh = pred_xywh_[index, :]

        score_ = np.reshape(score, (-1,))
        max_score = np.max(score_)
        bbox = np.concatenate([bbox_xywh[:2] - bbox_xywh[2:] / 2., bbox_xywh[:2] + bbox_xywh[2:] / 2.])
        bbox_xywh[2:] = self.size * (1 - lr) + bbox_xywh[2:] * lr
        self.center_pos = bbox_xywh[:2]
        self.size = bbox_xywh[2:]

        if self.success >= 0:
            out_score = score.max()
            if max_score < 0.7:
                self.success -= 1
                self.center_pos_lost = self.center_pos
                self.size_lost = self.size
                self.s_x_lost = s_x
                self.lost_frame = 0
            else:
                self.success = 1

        else:
            search_patches, offset_, scale_, num = build_detect_area(img, self.lost_frame, self.center_pos_lost,
                                                                     self.s_x_lost, self.search_size, max_num=64)
            zf_v = []
            for i in range(len(self.zf_v)):
                zf_v_ = self.zf_v[i].copy()
                zf_v_ = np.tile(zf_v_, (num, 1, 1))
                zf_v.append(zf_v_)
            score_, pred_lrtb_ = self.sess.run([self.score, self.bbox], feed_dict={self.x: search_patches,
                                                                                   self.zf_t[0]: zf_v[0],
                                                                                   self.zf_t[1]: zf_v[1],
                                                                                   self.zf_t[2]: zf_v[2],
                                                                                   self.zf_t[3]: zf_v[3],
                                                                                   self.zf_t[4]: zf_v[4],
                                                                                   self.zf_t[5]: zf_v[5],
                                                                                   K.learning_phase(): 0})
            pred_corner_ = pred_lrtb_ * scale_
            pred_corner_ = pred_corner_ + offset_
            pred_xywh_ = corner2center(pred_corner_)
            max_ = score_.max(axis=(1, 2))
            out_score = max_.max()
            if out_score > 0.95:
                index = max_.argmax()
                max_search_score = score_[index]
                max_search_score = np.reshape(max_search_score, (-1,))
                max_pred_xywh_ = np.reshape(pred_xywh_[index], (-1, 4))
                index = np.argmax(max_search_score)
                bbox_xywh = max_pred_xywh_[index, :]

                bbox = np.concatenate([bbox_xywh[:2] - bbox_xywh[2:] / 2., bbox_xywh[:2] + bbox_xywh[2:] / 2.])
                self.center_pos = bbox_xywh[:2]
                self.size = bbox_xywh[2:]
                self.success = 1
                self.lost_frame = 0
            else:
                self.lost_frame += 1

        if self.video_name is not None:
            box = list(map(int, bbox))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow(self.video_name, img)
            k = cv2.waitKey(15) & 0xff
            return bbox, bbox_xywh, k
        else:
            outputs = {'bbox': [bbox[0], bbox[1], bbox_xywh[2], bbox_xywh[3]],
                       'best_score': max(out_score, max_score)}
            return outputs

    def track_nfs(self, img, track_cfg):
        context_amount = track_cfg.context_amount
        penalty_k = track_cfg.penalty_k
        window_influence = track_cfg.window_influence
        lr = track_cfg.size_lr

        s_xs = []
        windows = []
        for ratio in self.ratios:
            s_x = round(np.sqrt(np.prod(self.size + context_amount * np.sum(self.size))) *
                        self.crop_settings_search['crop_size_rate'] * ratio)

            if s_x < self.s_x0 * 0.25:
                if s_x < self.s_x0 * 0.10:
                    s_x = self.s_x0 * 0.15
                else:
                    s_x = s_x * 1.12

            window_ = get_subwindow(img, self.center_pos, self.search_size, s_x, self.channel_average)
            windows.append(window_)
            s_xs.append(s_x)
        windows = np.concatenate(windows, axis=0)
        x_crop = rgb_normalize(windows, mobilenet=False)

        s_xs = np.array(s_xs)
        scales = s_xs / self.search_size  # (n, )
        scales = scales[:, None, None, None]
        s_xs = s_xs.reshape((-1, 1))
        offset = np.floor(self.center_pos.reshape((1, -1)) - s_xs / 2.)
        offset = np.concatenate([offset, offset], axis=-1)[:, None, None, :]  # (n, 32, 32, 4)

        score, pred_lrtb = self.sess.run([self.score, self.bbox], feed_dict={self.x: x_crop,
                                                                             self.zf_t[0]: self.zf_v_[0],
                                                                             self.zf_t[1]: self.zf_v_[1],
                                                                             self.zf_t[2]: self.zf_v_[2],
                                                                             self.zf_t[3]: self.zf_v_[3],
                                                                             self.zf_t[4]: self.zf_v_[4],
                                                                             self.zf_t[5]: self.zf_v_[5],
                                                                             K.learning_phase(): 0})

        pred_corner = pred_lrtb * scales
        pred_corner = pred_corner + offset
        pred_xywh = corner2center(pred_corner)
        pred_xywh = clip_bbox_center(pred_xywh, self.image_shape)

        s_c = change(sz(pred_xywh[..., 2], pred_xywh[..., 3], context_amount=context_amount) /
                     (sz(self.size[0], self.size[1], context_amount=context_amount)))
        r_c = change((self.size[0] / self.size[1]) / (pred_xywh[..., 2] / pred_xywh[..., 3]))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score
        wscore = (1. - window_influence) * pscore + window_influence * self.window

        wscore_ = np.reshape(wscore, (-1,))
        index = np.argmax(wscore_)
        pred_xywh_ = np.reshape(pred_xywh, (-1, 4))
        bbox_xywh = pred_xywh_[index, :]

        bbox = np.concatenate([bbox_xywh[:2] - bbox_xywh[2:] / 2., bbox_xywh[:2] + bbox_xywh[2:] / 2.])
        bbox_xywh[2:] = self.size * (1 - lr) + bbox_xywh[2:] * lr
        self.center_pos = bbox_xywh[:2]
        self.size = bbox_xywh[2:]

        if self.video_name is not None:
            box = list(map(int, bbox))
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow(self.video_name, img)
            k = cv2.waitKey(30) & 0xff
            return bbox, bbox_xywh, k
        else:
            score_ = score.reshape((-1,))
            outputs = {'bbox': [bbox[0], bbox[1], bbox_xywh[2], bbox_xywh[3]],
                       'best_score': score_[index]}
            return outputs


def build_detect_area(img, lost_frame, lost_center, s_x, search_size, max_num=64):
    sx_ratio = np.concatenate([np.logspace(np.log10(1.0), np.log10(1.04), num=15),
                               np.logspace(np.log10(1.04), np.log10(1.08), num=20),
                               np.logspace(np.log10(1.08), np.log10(1.12), num=20),
                               np.logspace(np.log10(1.12), np.log10(1.15), num=35)])
    search_ratio = np.concatenate([np.logspace(np.log10(2.), np.log10(3.), num=15),
                                   np.logspace(np.log10(3.), np.log10(4.), num=20),
                                   np.logspace(np.log10(4.), np.log10(5.), num=20),
                                   np.logspace(np.log10(5.), np.log10(8.), num=35)])
    num_patches = [5, 6, 7, 8]
    ih, iw = img.shape[:2]
    img_area = np.array([5., 5., iw - 5., ih - 5.], dtype=np.float32)
    ix1, iy1, ix2, iy2 = img_area

    """
    随着丢失时间的增长，不断扩大搜索范围，直到在整幅图像中进行检测搜索
    """
    if lost_frame <= 15:
        num_patch = num_patches[0]
    elif lost_frame <= 35:
        num_patch = num_patches[1]
    elif lost_frame <= 55:
        num_patch = num_patches[2]
    elif lost_frame <= 90:
        num_patch = num_patches[3]
    else:
        num_patch = None

    if lost_frame >= 90:
        search_area = img_area
        s_x = s_x * sx_ratio[-1]
    else:
        s_x = s_x * sx_ratio[lost_frame]
        S_X = s_x * search_ratio[lost_frame]
        search_area_xywh = np.array([lost_center[0], lost_center[1], S_X, S_X], dtype=np.float32)
        x1, y1, x2, y2 = center2corner(search_area_xywh)
        x1_, y1_, x2_, y2_ = x1, y1, x2, y2
        # 当检测区域有超过图像边界时，在内部进行一定补偿
        if x1 < ix1 and x2 < ix2:
            x2_ = x2 + (ix1 - x1) * 0.4
        if x1 > ix1 and x2 > ix2:
            x1_ = x1 + (x2 - ix2) * 0.4
        if y1 < iy1 and y2 < iy2:
            y2_ = y2 + (iy1 - y1) * 0.4
        if y1 > iy1 and y2 > iy2:
            y1_ = y1 + (y2 - iy2) * 0.4
        search_area_ = np.array([x1_, y1_, x2_, y2_], dtype=np.float32)
        search_area = np.concatenate \
            ([np.maximum(search_area_[:2], img_area[:2]), np.minimum(search_area_[2:], img_area[2:])])

    return create_search_patches(img, s_x, search_area, search_size, num_patch=num_patch, max_num=max_num)


def find_center_pos(length, s_x, num=None):
    if num is None:
        num = np.ceil(length / s_x)
    center = np.linspace(start=s_x / 2 + 5., stop=length - s_x / 2 - 5., num=int(num))
    return center


def fix_temp_area(bbox):
    ratio = max(bbox[2], bbox[3]) / min(bbox[2], bbox[3])
    area = bbox[2] * bbox[3]
    rate = 1.0
    if ratio < 1.05:
        if 250 ** 2 <= area < 270 ** 2:
            rate = 0.6
    elif ratio <= 1.16:
        if area < 30 ** 2:
            rate = 0.6
    elif 1.30 <= ratio < 1.50:
        if area >= 100 ** 2:
            rate = 0.65
    elif 1.80 <= ratio < 2.20:
        if area < 100 ** 2:
            rate = 0.6
    elif 2.20 <= ratio <= 2.70:
        if area >= 100 ** 2:
            rate = 0.6
    return rate


def create_search_patches(img, s_x, search_area, search_size, num_patch, max_num):
    x_center = find_center_pos(search_area[2] - search_area[0], s_x, num_patch) + search_area[0]
    y_center = find_center_pos(search_area[3] - search_area[1], s_x, num_patch) + search_area[1]
    x_center, y_center = np.meshgrid(x_center, y_center)
    center = np.concatenate((x_center[..., None], y_center[..., None]), axis=-1)
    center = np.reshape(center, (-1, 2))

    num = center.shape[0]
    if num > max_num:
        slt = np.arange(num)
        random.shuffle(slt)
        slt = slt[:max_num]
        center = center[slt]
        num = max_num

    size = np.ones((num, 2)) * s_x
    box_center = np.concatenate([center, size], axis=-1)
    box = center2corner(box_center).astype(np.int32)
    box = clip_bbox_corner(box, img.shape[:2])
    search_patches_ = []
    for i in range(num):
        search_patch = img[int(box[i, 1]): int(box[i, 3]), int(box[i, 0]): int(box[i, 2]), :]
        search_patch = cv2.resize(search_patch, (search_size, search_size), interpolation=cv2.INTER_CUBIC)
        search_patches_.append(search_patch)
    search_patches = np.array(search_patches_).astype(np.float32)
    search_patches = rgb_normalize(search_patches, mobilenet=False)

    scale = s_x / search_size
    offset = np.concatenate([box[:, :2], box[:, :2]], axis=-1)[:, None, None, :]
    return search_patches, offset, scale, num
