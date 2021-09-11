

import numpy as np
from keras.utils import Sequence

from training.Augmentation import random_sys, rand, random_crop
from training.Augmentation import image_augmentation, gray_aug, mix_aug, occ_aug, random_background
from utils.image import rgb_normalize
from utils.bbox import box2roi

import math
import random


class Generator(Sequence):
    def __init__(self,
                 validate,
                 dataloader,
                 encoder,
                 loss_num,
                 batch_size,
                 search_size,
                 template_size,
                 score_size,
                 crop_settings,
                 aug_settings,
                 encode_settings,
                 use_all_boxes=False,
                 use_z_box=False,
                 use_x_box=False):
        self.validate = validate
        self.encoder = encoder
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.loss_target = []
        for i in range(loss_num):
            self.loss_target.append(np.zeros(self.batch_size, dtype=np.float32))

        if self.validate:
            self.data_num = self.dataloader.num_val
        else:
            self.data_num = self.dataloader.num_train

        self.search_size = search_size
        self.template_size = template_size
        self.score_size = score_size
        self.crop_settings = crop_settings
        self.aug_settings = aug_settings
        self.encode_settings = encode_settings
        self.use_all_boxes = use_all_boxes
        self.use_z_box = use_z_box
        self.use_x_box = use_x_box

    def __len__(self):
        return math.floor(self.data_num // float(self.batch_size))

    def on_epoch_end(self):
        if not self.validate:
            self.dataloader.build_train_index()

    def __getitem__(self, index):
        # 根据分发的索引，生成每个batch数据
        temp_data = []
        temp_boxes = []
        search_data = []
        boxes = []
        mix_boxes = []

        neg = random_sys(self.batch_size)
        # 最简单负样本对，x和z分别从不同的序列中提取
        pos = neg > self.aug_settings['neg_pair']

        for b in range(self.batch_size):
            idx = index * self.batch_size + b

            # 从数据集中读入训练数据：
            # 搜索图像、模板图像、搜索图像上目标的 bbox、模板图像上的 bbox
            search_image, template_image = None, None
            while search_image is None or template_image is None:
                all_boxes, search_image, search_bbox, template_image, template_bbox = self.dataloader.read(idx, validate=self.validate, positive=pos[b])
                idx = random.randint(0, self.data_num-1)

            # 训练时，进行随机放缩与crop
            # crop 图像后，会 keep w/h 的将数据 resize with padding 到 model 要求的输入尺寸下
            template_img, template_box = random_crop(template_image, template_bbox, self.template_size, self.crop_settings['template'])
            search_img, search_box, all_boxes_ = random_crop(search_image, search_bbox, self.search_size, self.crop_settings['search'], all_boxes)

            mix_boxes.append(None)
            if all_boxes_ is not None:
                mix_boxes[-1] = all_boxes_

            # 只在训练时对图像进行扩增，验证时不扩增
            if not self.validate:

                # 正样本对
                if neg[b] > self.aug_settings['neg_threshold']:

                    # 随机将目标移动
                    translation = random_sys()
                    if translation < self.aug_settings['translation_background']:
                        scale_w, scale_h = rand(1.15, 1.35), rand(1.15, 1.35)
                        search_area = box2roi(search_box, scale_w, scale_h, self.search_size)
                        box = list(map(int, search_area))
                        target = search_img[box[1]: box[3] + 1, box[0]: box[2] + 1]

                        # 将目标（带一定背景区域）移动到随机选取的其他图像上
                        if translation < self.aug_settings['translation_other']:
                            other_image = None
                            while other_image is None:
                                # 随机找一副图像
                                other_mixes, other_image, other_bbox = self.dataloader.get_random_data(read_all_boxes=False)
                            other_img, other_box, other_mixes_ = random_crop(other_image, other_bbox, self.search_size, self.crop_settings['search'], other_mixes)
                            # 将目标区域（带一定大小的背景）移动到随机抽取的图像上，之前的目标仍为正，该图像上的所有物体均为负
                            search_img_, target_box = mix_aug(other_img, other_box, target, min_rate=0.8, max_rate=2.0)
                            other_box = other_box.reshape((-1, 4))
                            if target_box is not None:
                                search_box = box2roi(target_box, 1. / scale_w, 1. / scale_h, self.search_size)
                                search_img = search_img_
                                if other_mixes_ is not None:
                                    mix_boxes[-1] = np.concatenate([other_box, other_mixes_])
                                else:
                                    mix_boxes[-1] = other_box

                        # 将目标（带一定背景区域）移动到图像中其他位置上
                        else:
                            bg_box = random_background(search_image, search_bbox, min_rate=0.8, max_rate=1.25,
                                                       crop_settings=self.crop_settings['val'])
                            if bg_box is not None:
                                bg_img, bg_box_, all_boxes_bg = random_crop(search_image, bg_box, self.search_size, self.crop_settings['val'], all_boxes)
                                # 将目标区域（带一定大小的背景）移动到图像的其他位置上，目标仍为正，无mix boxes
                                search_img_, target_box = mix_aug(bg_img, bg_box_, target, min_rate=0.2, max_rate=2.0)
                                if target_box is not None:
                                    search_box = box2roi(target_box, 1. / scale_w, 1. / scale_h, self.search_size)
                                    search_img = search_img_
                                    mix_boxes[-1] = all_boxes_bg
                else:
                    overlap_thresh = 0.8
                    # 人为制造full occlusion导致的负样本对
                    if self.aug_settings['neg_pair'] <= neg[b] < self.aug_settings['occ_object']:
                        # 在当前图像中找到一块背景区域作为遮挡块
                        if neg[b] < self.aug_settings['occ_background']:
                            occ_image = search_image
                            occ_box = random_background(occ_image, search_bbox,
                                                        crop_settings=rand(-0.1, 0.25), min_rate=0.8, max_rate=1.25)
                        # 从其他序列中crop出物体作为遮挡块
                        else:
                            occ_image = None
                            while occ_image is None:
                                _, occ_image, occ_box = self.dataloader.get_random_data(read_all_boxes=False)
                        if occ_box is not None:
                            occ_image_, occ_box_ = random_crop(occ_image, occ_box, self.search_size, self.crop_settings['search'])
                            occ_box_ = box2roi(occ_box_, rand(1.15, 1.35), rand(1.15, 1.35), self.search_size)
                            box = list(map(int, occ_box_))
                            occ = occ_image_[box[1]: box[3] + 1, box[0]: box[2] + 1]
                            search_img_, occed_box, overlap = occ_aug(search_img, search_box, occ, .15,
                                                                      try_num=5, overlap_thresh=overlap_thresh)
                            # 当目标大部分被完全覆盖时，才完成有效的遮挡扩增
                            if occed_box is not None and overlap > overlap_thresh:
                                search_img = search_img_
                                pos[b] = 0

                # 当图像中只有目标这一个框时，加入干扰物体
                if mix_boxes[-1] is None and random_sys() < self.aug_settings['mix']:
                    num_mix = random.randint(1, 2)
                    mix_boxes_ = []
                    counter = 0
                    for i in range(num_mix):
                        mix_image = None
                        while mix_image is None:
                            _, mix_image, mix_box = self.dataloader.get_random_data(read_all_boxes=False)
                        mix_image_, mix_box_ = random_crop(mix_image, mix_box, self.search_size, self.crop_settings['search'])
                        mix_box_ = box2roi(mix_box_, rand(1.15, 1.35), rand(1.15, 1.35), self.search_size)
                        box = list(map(int, mix_box_))
                        mix = mix_image_[box[1]: box[3] + 1, box[0]: box[2] + 1]
                        min_rate, max_rate = 0.8, 2.0
                        search_img, mixed_box = mix_aug(search_img, search_box, mix, min_rate, max_rate)
                        if mixed_box is not None:
                            mix_boxes_.append(mixed_box)
                            counter += 1
                    if counter > 0:
                        mix_boxes_ = np.array(mix_boxes_).reshape((-1, 4))
                        mix_boxes[-1] = mix_boxes_

                template_img, template_box = image_augmentation(template_img, template_box, self.aug_settings['template'])
                # 负样本对时，不进行普通扩增
                if pos[b]:
                    if mix_boxes[-1] is None:
                        search_img, search_box = image_augmentation(search_img, search_box, self.aug_settings['search'])
                    else:
                        search_img, search_box, mix_boxes[-1] = image_augmentation(search_img, search_box,
                                                                                   self.aug_settings['search'],
                                                                                   mix_boxes[-1])

                if self.aug_settings['gray'] and random_sys() < self.aug_settings['gray']:
                    template_img = gray_aug(template_img)
                    search_img = gray_aug(search_img)
            temp_data.append(template_img)
            temp_boxes.append(template_box)
            search_data.append(search_img)
            boxes.append(search_box)

        boxes = np.array(boxes).astype(np.float32)
        temp_boxes = np.array(temp_boxes).astype(np.float32)
        temp_data = np.array(temp_data)
        search_data = np.array(search_data)

        # 对图像进行归一化，从 BGR 转 RGB 格式，并按照 EfficientNet 训练时的归一化方式进行归一化
        temp_data = rgb_normalize(temp_data)
        search_data = rgb_normalize(search_data)

        # 最后，根据 bbox 进行encode，生成label
        if self.use_all_boxes:
            labels = self.encoder(boxes, mix_boxes, pos, self.search_size, self.score_size, **self.encode_settings)
        else:
            labels = self.encoder(boxes, pos, self.search_size, self.score_size, **self.encode_settings)

        inputs = [search_data, temp_data]
        if self.use_z_box:
            inputs.append(temp_boxes)
        if self.use_x_box:
            inputs.append(boxes)
        return tuple([inputs + list(labels), self.loss_target])

    def debug(self, index):
        import cv2

        # 根据分发的索引，生成每个batch数据
        temp_data = []
        temp_boxes = []
        search_data = []
        boxes = []
        mix_boxes = []

        images = []
        augs = []

        neg = random_sys(self.batch_size)
        # 最简单负样本对，x和z分别从不同的序列中提取
        pos = neg > self.aug_settings['neg_pair']

        for b in range(self.batch_size):
            aug = []
            if not pos[b]:
                aug.append('neg')
            idx = index * self.batch_size + b

            # 从数据集中读入训练数据：
            # 搜索图像、模板图像、搜索图像上目标的 bbox、模板图像上的 bbox
            search_image, template_image = None, None
            while search_image is None or template_image is None:
                all_boxes, search_image, search_bbox, template_image, template_bbox = self.dataloader.read(idx, validate=self.validate, positive=pos[b])
                idx = random.randint(0, self.data_num-1)

            template_ = template_image.astype(np.uint8)
            box = list(map(int, template_bbox))
            cv2.rectangle(template_, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            search_ = search_image.astype(np.uint8)
            box = list(map(int, search_bbox))
            cv2.rectangle(search_, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            if len(all_boxes) > 0:
                for i in range(len(all_boxes)):
                    box = list(map(int, all_boxes[i]))
                    cv2.rectangle(search_, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # cv2.imshow('01', template_)
            # cv2.imshow('02', search_)
            # cv2.waitKey()

            # 训练时，进行随机放缩与crop
            # crop 图像后，会 keep w/h 的将数据 resize with padding 到 model 要求的输入尺寸下
            template_img, template_box = random_crop(template_image, template_bbox, self.template_size, self.crop_settings['template'])
            search_img, search_box, all_boxes_ = random_crop(search_image, search_bbox, self.search_size, self.crop_settings['search'], all_boxes)

            template0 = template_img.astype(np.uint8)
            box = list(map(int, template_box))
            cv2.rectangle(template0, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            search0 = search_img.astype(np.uint8)
            box = list(map(int, search_box))
            cv2.rectangle(search0, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            if all_boxes_ is not None:
                for i in range(all_boxes_.shape[0]):
                    box = list(map(int, all_boxes_[i, :]))
                    cv2.rectangle(search0, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # cv2.imshow('1', template0)
            # cv2.imshow('2', search0)
            # cv2.waitKey()

            mix_boxes.append(None)
            if all_boxes_ is not None:
                mix_boxes[-1] = all_boxes_

            # 只在训练时对图像进行扩增，验证时不扩增
            if not self.validate:

                # 正样本对
                if neg[b] > self.aug_settings['neg_threshold']:

                    # 随机将目标移动
                    translation = random_sys()
                    if translation < self.aug_settings['translation_background']:
                        scale_w, scale_h = rand(1.15, 1.35), rand(1.15, 1.35)
                        search_area = box2roi(search_box, scale_w, scale_h, self.search_size)
                        box = list(map(int, search_area))
                        target = search_img[box[1]: box[3] + 1, box[0]: box[2] + 1]

                        # 将目标（带一定背景区域）移动到随机选取的其他图像上
                        if translation < self.aug_settings['translation_other']:
                            other_image = None
                            while other_image is None:
                                # 随机找一副图像
                                other_mixes, other_image, other_bbox = self.dataloader.get_random_data(read_all_boxes=False)
                            other_img, other_box, other_mixes_ = random_crop(other_image, other_bbox, self.search_size, self.crop_settings['search'], other_mixes)
                            # 将目标区域（带一定大小的背景）移动到随机抽取的图像上，之前的目标仍为正，该图像上的所有物体均为负
                            search_img_, target_box = mix_aug(other_img, other_box, target, min_rate=0.8, max_rate=2.0)
                            other_box = other_box.reshape((-1, 4))
                            if target_box is not None:
                                search_box = box2roi(target_box, 1. / scale_w, 1. / scale_h, self.search_size)
                                search_img = search_img_
                                if other_mixes_ is not None:
                                    mix_boxes[-1] = np.concatenate([other_box, other_mixes_])
                                else:
                                    mix_boxes[-1] = other_box
                                aug.append('translation other')

                        # 将目标（带一定背景区域）移动到图像中其他位置上
                        else:
                            bg_box = random_background(search_image, search_bbox, min_rate=0.8, max_rate=1.25,
                                                       crop_settings=self.crop_settings['val'])
                            if bg_box is not None:
                                bg_img, bg_box_, all_boxes_bg = random_crop(search_image, bg_box, self.search_size, self.crop_settings['val'], all_boxes)
                                # 将目标区域（带一定大小的背景）移动到图像的其他位置上，目标仍为正，无mix boxes
                                search_img_, target_box = mix_aug(bg_img, bg_box_, target, min_rate=0.2, max_rate=2.0)
                                if target_box is not None:
                                    search_box = box2roi(target_box, 1. / scale_w, 1. / scale_h, self.search_size)
                                    search_img = search_img_
                                    mix_boxes[-1] = all_boxes_bg

                                    aug = ['translation background']
                                    search_bg = search_image.astype(np.uint8)
                                    box = list(map(int, search_bbox))
                                    cv2.rectangle(search_bg, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                                    if len(all_boxes) > 0:
                                        for i in range(len(all_boxes)):
                                            box = list(map(int, all_boxes[i]))
                                            cv2.rectangle(search_bg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                                    box = list(map(int, bg_box))
                                    cv2.rectangle(search_bg, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                                    # cv2.imshow('bg0', search_bg)
                                    # cv2.waitKey()

                                    bg = bg_img.astype(np.uint8)
                                    box = list(map(int, bg_box_))
                                    cv2.rectangle(bg, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                                    box = list(map(int, search_box))
                                    cv2.rectangle(bg, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                                    if all_boxes_bg is not None:
                                        for i in range(len(all_boxes_bg)):
                                            box = list(map(int, all_boxes_bg[i]))
                                            cv2.rectangle(bg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                                    # cv2.imshow('bg', bg)
                                    # cv2.waitKey()
                else:
                    overlap_thresh = 0.8
                    # 人为制造full occlusion导致的负样本对
                    if self.aug_settings['neg_pair'] <= neg[b] < self.aug_settings['occ_object']:
                        # 在当前图像中找到一块背景区域作为遮挡块
                        if neg[b] < self.aug_settings['occ_background']:
                            occ_image = search_image
                            occ_box = random_background(occ_image, search_bbox,
                                                        crop_settings=rand(-0.1, 0.25), min_rate=0.8, max_rate=1.25)
                        # 从其他序列中crop出物体作为遮挡块
                        else:
                            occ_image = None
                            while occ_image is None:
                                _, occ_image, occ_box = self.dataloader.get_random_data(read_all_boxes=False)
                        if occ_box is not None:
                            occ_image_, occ_box_ = random_crop(occ_image, occ_box, self.search_size, self.crop_settings['search'])
                            occ_box_ = box2roi(occ_box_, rand(1.15, 1.35), rand(1.15, 1.35), self.search_size)
                            box = list(map(int, occ_box_))
                            occ = occ_image_[box[1]: box[3] + 1, box[0]: box[2] + 1]
                            search_img_, occed_box, overlap = occ_aug(search_img, search_box, occ, .15,
                                                                      try_num=5, overlap_thresh=overlap_thresh)
                            # 当目标大部分被完全覆盖时，才完成有效的遮挡扩增
                            if occed_box is not None and overlap > overlap_thresh:
                                search_img = search_img_
                                pos[b] = 0
                                if neg[b] < self.aug_settings['occ_background']:
                                    aug.append('occ background')
                                else:
                                    aug.append('occ other')

                # 当图像中只有目标这一个框时，加入干扰物体
                if mix_boxes[-1] is None and random_sys() < self.aug_settings['mix']:
                    num_mix = random.randint(1, 2)
                    mix_boxes_ = []
                    counter = 0
                    for i in range(num_mix):
                        mix_image = None
                        while mix_image is None:
                            _, mix_image, mix_box = self.dataloader.get_random_data(read_all_boxes=False)
                        mix_image_, mix_box_ = random_crop(mix_image, mix_box, self.search_size, self.crop_settings['search'])
                        mix_box_ = box2roi(mix_box_, rand(1.15, 1.35), rand(1.15, 1.35), self.search_size)
                        box = list(map(int, mix_box_))
                        mix = mix_image_[box[1]: box[3] + 1, box[0]: box[2] + 1]
                        min_rate, max_rate = 0.8, 2.0
                        search_img, mixed_box = mix_aug(search_img, search_box, mix, min_rate, max_rate)
                        if mixed_box is not None:
                            mix_boxes_.append(mixed_box)
                            counter += 1
                    if counter > 0:
                        mix_boxes_ = np.array(mix_boxes_).reshape((-1, 4))
                        mix_boxes[-1] = mix_boxes_
                        aug.append('mix')

                template_img, template_box = image_augmentation(template_img, template_box, self.aug_settings['template'])
                # 负样本对时，不进行普通扩增
                if pos[b]:
                    if mix_boxes[-1] is None:
                        search_img, search_box = image_augmentation(search_img, search_box, self.aug_settings['search'])
                    else:
                        search_img, search_box, mix_boxes[-1] = image_augmentation(search_img, search_box,
                                                                                   self.aug_settings['search'],
                                                                                   mix_boxes[-1])

                if self.aug_settings['gray'] and random_sys() < self.aug_settings['gray']:
                    template_img = gray_aug(template_img)
                    search_img = gray_aug(search_img)

            template1 = template_img.astype(np.uint8)
            box = list(map(int, template_box))
            cv2.rectangle(template1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            search1 = search_img.astype(np.uint8)
            box = list(map(int, search_box))
            cv2.rectangle(search1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            if mix_boxes[-1] is not None:
                for i in range(mix_boxes[-1].shape[0]):
                    box = list(map(int, mix_boxes[-1][i, :]))
                    cv2.rectangle(search1, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # cv2.imshow('3', template1)
            # cv2.imshow('4', search1)
            # cv2.waitKey()
            augs.append(aug)
            temp_data.append(template_img)
            temp_boxes.append(template_box)
            search_data.append(search_img)
            boxes.append(search_box)
            images.append([template_, search_, template0, search0, template1, search1])

        boxes = np.array(boxes).astype(np.float32)
        temp_boxes = np.array(temp_boxes).astype(np.float32)
        temp_data = np.array(temp_data)
        search_data = np.array(search_data)

        # 对图像进行归一化，从 BGR 转 RGB 格式，并按照 EfficientNet 训练时的归一化方式进行归一化
        temp_data = rgb_normalize(temp_data)
        search_data = rgb_normalize(search_data)

        # 最后，根据 bbox 进行encode，生成label
        if self.use_all_boxes:
            labels = self.encoder(boxes, mix_boxes, pos, self.search_size, self.score_size, **self.encode_settings)
        else:
            labels = self.encoder(boxes, pos, self.search_size, self.score_size, **self.encode_settings)

        inputs = [search_data, temp_data]
        if self.use_z_box:
            inputs.append(temp_boxes)
        if self.use_x_box:
            inputs.append(boxes)

        return tuple([inputs + list(labels), self.loss_target]), images, boxes, temp_boxes, pos, augs
