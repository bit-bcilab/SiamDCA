

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

from configs.base_config import base_cfg, TRAIN_SETTINGS
from utils.grid import generate_grid
from model.Backbone import PRETRAINED_BACKBONE_ROOT, PRETRAINED_BACKBONE_PATH, BACKBONE_OUT_LAYERS

from model.Models import Build_SiamDCA
from tracker.BoxDecoder import LTRBDecoder
from training.BoxEncoder import ltrb_mix_encoder
from training.Loss import NllNegLoss, NllPosLoss, SmoothL1Loss, CIOULoss, PreprocessCls

from training.DataLoader import DataLoader
from training.Generator import Generator
from training.CallBacks import LRWarmUpStepDeCayRestart, SaveUnfrozenModel, SaveUnfrozenModel_SingleGPU
from utils.keras_model import load_and_freeze, multi_gpu_model

import yaml
import argparse
import os

parser = argparse.ArgumentParser(description='SiamDCA Training')
parser.add_argument('--tracker', default='SiamDCA', type=str, help='name of the tracker')
parser.add_argument('--log_dir', default='SiamDCA', type=str, help='path of checkpoint')
parser.add_argument('--num_gpu', default=2, type=int, help="number of gpus")
parser.add_argument('--gpu_id', default='0, 1', type=str, help="gpu id")
parser.add_argument('--num_worker', default=16, type=int, help="number of workers")
parser.add_argument('--max_queue_size', default=32, type=int, help="max number of queues")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def train_SiamDCA():
    log_dir = os.path.join('logs', args.log_dir)
    base_cfg.merge_from_file(os.path.join('configs', args.tracker, 'base.yaml'))
    # yaml file which saves the hyper parameters for training
    with open(os.path.join('configs', args.tracker, 'train_settings.yaml'), 'r', encoding='utf-8') as f:
        cond = f.read()
        train_settings = yaml.load(cond)
        f.close()
    TRAIN_SETTINGS.update(train_settings)

    batch = int(args.num_gpu * base_cfg.TRAIN.BATCH_PER_GPU)
    num_loss = len(TRAIN_SETTINGS['LOSS_WEIGHT'])

    dataloader = DataLoader(data_settings=TRAIN_SETTINGS['DATA_SETTINGS'], read_all_boxes=True)
    grid = generate_grid(search_size=base_cfg.SEARCH_SIZE[:-1], score_size=base_cfg.MODEL.SCORE_SIZE)
    TRAIN_SETTINGS['ENCODE_SETTINGS'].update(dict(grid=grid))
    data_kwargs = dict(dataloader=dataloader,
                       encoder=ltrb_mix_encoder,
                       batch_size=batch,
                       loss_num=num_loss,
                       search_size=base_cfg.SEARCH_SIZE[:-1],
                       template_size=base_cfg.TEMP_SIZE[:-1],
                       score_size=base_cfg.MODEL.SCORE_SIZE,
                       crop_settings=TRAIN_SETTINGS['CROP_SETTINGS'],
                       aug_settings=TRAIN_SETTINGS['AUG_SETTINGS'],
                       encode_settings=TRAIN_SETTINGS['ENCODE_SETTINGS'],
                       use_z_box=False,
                       use_x_box=False,
                       use_all_boxes=True)
    train_generator = Generator(validate=False, **data_kwargs)
    val_generator = Generator(validate=True, **data_kwargs)

    model = Build_SiamDCA(choice=base_cfg.MODEL.CHOICE, fpn_in=base_cfg.MODEL.FPN_IN,
                          x_shape=[batch] + base_cfg.SEARCH_SIZE, z_shape=[batch] + base_cfg.TEMP_SIZE,
                          num_filters=base_cfg.MODEL.NUM_FILTERS, num_att=base_cfg.MODEL.NUM_ATT,
                          num_att_heads=base_cfg.MODEL.NUM_ATT_HEADS, d_k=base_cfg.MODEL.D_K, d_v=base_cfg.MODEL.D_V,
                          hidden_dims=base_cfg.MODEL.HIDDEN_DIMS, dropout=base_cfg.MODEL.DROPOUT)

    model_train, loss_dict = build_train_model(model, grid=grid, batch=batch, score_size=base_cfg.MODEL.SCORE_SIZE)

    model = load_and_freeze(model, weight_path='backbone-only', frozen_layers=[['backbone', 'all']],
                            pretrained_path=PRETRAINED_BACKBONE_ROOT + PRETRAINED_BACKBONE_PATH[base_cfg.MODEL.CHOICE])

    opt_kwargs = dict(optimizer=SGD,
                      optimizer_kwargs=dict(lr=base_cfg.TRAIN.INIT_LR, momentum=base_cfg.TRAIN.MOMENTUM),
                      compile_kwargs=dict(loss=loss_dict, loss_weights=TRAIN_SETTINGS['LOSS_WEIGHT']))
    opt_kwargs['compile_kwargs'].update(dict(optimizer=opt_kwargs['optimizer'](**opt_kwargs['optimizer_kwargs'])))

    logging = TensorBoard(log_dir=log_dir, update_freq=100)

    steps_per_epoch = int(dataloader.num_train / batch)
    unfreeze_step = int(steps_per_epoch * (base_cfg.TRAIN.WARM_UP +
                                           base_cfg.TRAIN.RESTART_EPOCHS * base_cfg.TRAIN.UNFREEZE_TIMES))
    lr_decay = LRWarmUpStepDeCayRestart(warm_up_steps=int(steps_per_epoch * base_cfg.TRAIN.WARM_UP),
                                        restart_steps=int(steps_per_epoch * base_cfg.TRAIN.RESTART_EPOCHS),
                                        restart_num=int(base_cfg.TRAIN.EPOCHS // base_cfg.TRAIN.RESTART_EPOCHS - 1),
                                        decay_steps=int(steps_per_epoch * base_cfg.TRAIN.DECAY_EPOCHS),
                                        decay_num=int(base_cfg.TRAIN.DECAY_EPOCHS / base_cfg.TRAIN.DECAY_STEPS * steps_per_epoch),
                                        init_lr=base_cfg.TRAIN.INIT_LR,
                                        stop_lr=base_cfg.TRAIN.MINI_LR,
                                        mid_lr=base_cfg.TRAIN.INIT_LR * base_cfg.TRAIN.MID_LR_RATIO,
                                        mid_step=int(steps_per_epoch * base_cfg.TRAIN.MID_EPOCHS),
                                        loss_num=num_loss,
                                        unfreeze_step=unfreeze_step,
                                        unfrozen_layers=BACKBONE_OUT_LAYERS[base_cfg.MODEL.CHOICE][-1],
                                        multi_gpu=(args.num_gpu > 1),
                                        opt_kwargs=opt_kwargs)

    if args.num_gpu > 1:
        saver = SaveUnfrozenModel(num_loss=num_loss, log_dir=log_dir, weights_name='epoch')
        model_gpu = multi_gpu_model(model_train, gpus=args.num_gpu)
        model_gpu.compile(**opt_kwargs['compile_kwargs'])
        model_gpu.fit_generator(
            generator=train_generator,
            workers=args.num_worker,
            use_multiprocessing=True,
            max_queue_size=args.max_queue_size,
            steps_per_epoch=max(1, steps_per_epoch),
            validation_data=val_generator,
            validation_steps=max(1, int(dataloader.num_val // batch)),
            epochs=base_cfg.TRAIN.EPOCHS,
            initial_epoch=0,
            callbacks=[logging, saver, lr_decay])
    else:
        saver = SaveUnfrozenModel_SingleGPU(log_dir=log_dir, weights_name='epoch')
        model_train.compile(**opt_kwargs['compile_kwargs'])
        model_train.fit_generator(
            generator=train_generator,
            workers=args.num_worker,
            use_multiprocessing=True,
            max_queue_size=args.max_queue_size,
            steps_per_epoch=max(1, steps_per_epoch),
            validation_data=val_generator,
            validation_steps=max(1, int(dataloader.num_val // batch)),
            epochs=base_cfg.TRAIN.EPOCHS,
            initial_epoch=0,
            callbacks=[logging, saver, lr_decay])


def build_train_model(model, grid, batch, score_size):
    label_loc = Input(batch_shape=(batch, score_size[0], score_size[1], 4))
    label_cls = Input(batch_shape=(batch, score_size[0], score_size[1]))
    pred_cls, pred_loc = model.output

    label_cls_, pred_cls_ = PreprocessCls(mode='log-softmax')([label_cls, pred_cls])
    neg_loss = NllNegLoss(name='neg')([label_cls_, pred_cls_])
    pos_loss = NllPosLoss(name='pos')([label_cls_, pred_cls_])

    mask = Lambda(lambda tensor: tf.cast(tf.equal(tensor, 1.), dtype=tf.float32))(label_cls)
    l1_loss = SmoothL1Loss(name='l1')([label_loc, pred_loc, mask])
    bbox = LTRBDecoder(grid=grid)(label_loc)
    pred_box = LTRBDecoder(grid=grid)(pred_loc)
    iou_loss, iou = CIOULoss(return_iou=True, name='iou')([bbox, pred_box, mask])

    model_train = Model([*model.input, label_cls, label_loc], [pos_loss, neg_loss, iou_loss, l1_loss])
    loss_dict = {'pos': lambda y_true, y_pred: y_pred,
                 'neg': lambda y_true, y_pred: y_pred,
                 'iou': lambda y_true, y_pred: y_pred,
                 'l1': lambda y_true, y_pred: y_pred}
    return model_train, loss_dict


def check_data():
    import numpy as np
    import cv2

    base_cfg.merge_from_file(os.path.join('configs', args.tracker, 'base.yaml'))
    with open(os.path.join('configs', args.tracker, 'train_settings.yaml'), 'r', encoding='utf-8') as f:
        cond = f.read()
        train_settings = yaml.load(cond)
        f.close()
    TRAIN_SETTINGS.update(train_settings)

    batch = int(args.num_gpu * base_cfg.TRAIN.BATCH_PER_GPU)
    num_loss = len(TRAIN_SETTINGS['LOSS_WEIGHT'])

    dataloader = DataLoader(data_settings=TRAIN_SETTINGS['DATA_SETTINGS'], read_all_boxes=True)

    for i in range(10000):
        all_boxes, search_img, search_box, template_img, template_box = dataloader.read(i, False, True)
        template0 = template_img.astype(np.uint8)
        box = list(map(int, template_box))
        cv2.rectangle(template0, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow('1', template0)
        search0 = search_img.astype(np.uint8)
        box = list(map(int, search_box))
        cv2.rectangle(search0, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        if len(all_boxes) > 0:
            for all_box in all_boxes:
                box = list(map(int, all_box))
                cv2.rectangle(search0, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imshow('2', search0)
        cv2.waitKey()

        all_boxes, search_img, search_box, template_img, template_box = dataloader.read(i, True, False)
        template0 = template_img.astype(np.uint8)
        box = list(map(int, template_box))
        cv2.rectangle(template0, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow('1', template0)
        search0 = search_img.astype(np.uint8)
        box = list(map(int, search_box))
        cv2.rectangle(search0, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        if len(all_boxes) > 0:
            for all_box in all_boxes:
                box = list(map(int, all_box))
                cv2.rectangle(search0, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imshow('2', search0)
        cv2.waitKey()

    grid = generate_grid(search_size=base_cfg.SEARCH_SIZE[:-1], score_size=base_cfg.MODEL.SCORE_SIZE)
    TRAIN_SETTINGS['ENCODE_SETTINGS'].update(dict(grid=grid))
    data_kwargs = dict(dataloader=dataloader,
                       encoder=ltrb_mix_encoder,
                       batch_size=batch,
                       loss_num=num_loss,
                       search_size=base_cfg.SEARCH_SIZE[:-1],
                       template_size=base_cfg.TEMP_SIZE[:-1],
                       score_size=base_cfg.MODEL.SCORE_SIZE,
                       crop_settings=TRAIN_SETTINGS['CROP_SETTINGS'],
                       aug_settings=TRAIN_SETTINGS['AUG_SETTINGS'],
                       encode_settings=TRAIN_SETTINGS['ENCODE_SETTINGS'],
                       use_z_box=False,
                       use_x_box=False,
                       use_all_boxes=True)
    train_generator = Generator(validate=False, **data_kwargs)
    val_generator = Generator(validate=True, **data_kwargs)
    for i in range(10000):
        if i % 2:
            a = val_generator.debug(i)
        else:
            a = train_generator.debug(i)
        # for j in range(batch):
        #     cv2.imshow('1', a[1][j][0])
        #     cv2.imshow('2', a[1][j][1])
        #     cv2.imshow('3', a[1][j][2])
        #     cv2.imshow('4', a[1][j][3])
        #     cv2.imshow('5', a[1][j][4])
        #     cv2.imshow('6', a[1][j][5])
        #     cv2.waitKey()

    model = Build_SiamDCA(choice=base_cfg.MODEL.CHOICE, fpn_in=base_cfg.MODEL.FPN_IN,
                          x_shape=[batch] + base_cfg.SEARCH_SIZE, z_shape=[batch] + base_cfg.TEMP_SIZE,
                          num_filters=base_cfg.MODEL.NUM_FILTERS, num_att=base_cfg.MODEL.NUM_ATT,
                          num_att_heads=base_cfg.MODEL.NUM_ATT_HEADS, d_k=base_cfg.MODEL.D_K, d_v=base_cfg.MODEL.D_V,
                          hidden_dims=base_cfg.MODEL.HIDDEN_DIMS, dropout=base_cfg.MODEL.DROPOUT)

    label_loc = Input(batch_shape=(batch, base_cfg.MODEL.SCORE_SIZE[0], base_cfg.MODEL.SCORE_SIZE[1], 4))
    label_cls = Input(batch_shape=(batch, base_cfg.MODEL.SCORE_SIZE[0], base_cfg.MODEL.SCORE_SIZE[1]))
    pred_cls, pred_loc = model.output
    label_cls_, pred_cls_ = PreprocessCls(mode='log-softmax')([label_cls, pred_cls])
    neg_loss = NllNegLoss(name='neg')([label_cls_, pred_cls_])
    pos_loss = NllPosLoss(name='pos')([label_cls_, pred_cls_])
    mask = Lambda(lambda tensor: tf.cast(tf.equal(tensor, 1.), dtype=tf.float32))(label_cls)
    l1_loss = SmoothL1Loss(name='l1')([label_loc, pred_loc, mask])
    bbox = LTRBDecoder(grid=grid)(label_loc)
    pred_box = LTRBDecoder(grid=grid)(pred_loc)
    iou_loss, iou = CIOULoss(return_iou=True, name='iou')([bbox, pred_box, mask])
    score = tf.nn.softmax(pred_cls, axis=-1)[..., 1]

    # model = load_and_freeze(model, weight_path='backbone-only', frozen_layers=[['backbone', 'all']],
    #                         pretrained_path=PRETRAINED_BACKBONE_ROOT + PRETRAINED_BACKBONE_PATH[base_cfg.MODEL.CHOICE])
    # model.load_weights('logs/SiamDCA/epoch48.h5', by_name=True)
    model.load_weights('weights/SiamDCA/SiamDCA.h5', by_name=True)
    for i in range(1000):
        if i % 2:
            a0 = val_generator.debug(i)
        else:
            a0 = train_generator.debug(i)
        a1 = sess.run([pred_box, pred_loc, pred_cls_, pred_cls,
                       neg_loss, pos_loss, label_cls_, label_cls, mask,
                       l1_loss, iou_loss, bbox, pred_box, iou, score],
                      feed_dict={model.input[0]: a0[0][0][0], model.input[1]: a0[0][0][1],
                                 label_cls: a0[0][0][2], label_loc: a0[0][0][3]})
        cv2.imshow('1', a0[1][3][0])
        cv2.imshow('2', a0[1][3][1])
        cv2.imshow('3', a0[1][3][2])
        cv2.imshow('4', a0[1][3][3])
        cv2.imshow('5', a0[1][3][4])
        cv2.imshow('6', a0[1][3][5])
        cv2.waitKey()


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    with tf.Session(config=config) as sess:
        set_session(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # check_data()
        train_SiamDCA()
