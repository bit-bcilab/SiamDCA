# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
from utils.bbox import get_axis_aligned_bbox
from toolkit.utils.region import vot_overlap, vot_float2str
from toolkit.utils import success_overlap, success_error

import cv2
import os


def convert_bb_to_center(bboxes):
    return np.array([(bboxes[:, 0] + (bboxes[:, 2] - 1) / 2),
                     (bboxes[:, 1] + (bboxes[:, 3] - 1) / 2)]).T


def convert_bb_to_norm_center(bboxes, gt_wh):
    return convert_bb_to_center(bboxes) / (gt_wh+1e-16)


def test(tracker, name, track_cfg, dataset, test_video='', save=True, visual=False):
    total_lost = 0
    if dataset.name in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if test_video != '':
                # test one special video
                if video.name != test_video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                               gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                               gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))[-1]
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_, video_name=None)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img, track_cfg=track_cfg)
                    pred_bbox = outputs['bbox']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5  # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic

                if idx == 0:
                    cv2.destroyAllWindows()
                if visual and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 3)
                    bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (bbox[0], bbox[1]),
                                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    k = cv2.waitKey(15) & 0xff
                    if k == 27:
                        cv2.destroyWindow(video.name)
                        break

            toc /= cv2.getTickFrequency()
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                v_idx + 1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number

            if visual and dataset.name != 'VOT2018-LT':
                tracker_traj = np.array(pred_bboxes[1:])
                gt_traj = np.array(video.gt_traj)
                n_frame = len(gt_traj)
                a_o = success_overlap(gt_traj[1:, :], tracker_traj, n_frame)
                thresholds = np.arange(0, 51, 1)
                gt_center = convert_bb_to_center(gt_traj)
                tracker_center = convert_bb_to_center(tracker_traj)
                a_p = success_error(gt_center[1:, :], tracker_center, thresholds, n_frame)
                print("precision: %.4f, AUC: %.4f" % (a_p[20], np.mean(a_o)))

            if save:
                # save results
                video_path = os.path.join('results', dataset.name, name, 'baseline', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        if isinstance(x, int):
                            f.write("{:d}\n".format(x))
                        else:
                            f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
        print("{:s} total lost: {:d}".format(name, total_lost))

    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if test_video != '':
                # test one special video
                if video.name != test_video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    # cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))[0]
                    w, h = gt_bbox[2:]
                    cx = gt_bbox[0] + w / 2
                    cy = gt_bbox[1] + h / 2
                    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                    tracker.init(img, gt_bbox_, video_name=None)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == dataset.name:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img, track_cfg=track_cfg)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if visual and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    k = cv2.waitKey(15) & 0xff
                    if k == 27:
                        cv2.destroyWindow(video.name)
                        break

            toc /= cv2.getTickFrequency()
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx + 1, video.name, toc, idx / toc))

            if visual and dataset.name != 'VOT2018-LT':
                tracker_traj = np.array(pred_bboxes)
                gt_traj = np.array(video.gt_traj)
                n_frame = len(gt_traj)
                a_o = success_overlap(gt_traj, tracker_traj, n_frame)
                thresholds = np.arange(0, 51, 1)
                gt_center = convert_bb_to_center(gt_traj)
                tracker_center = convert_bb_to_center(tracker_traj)
                a_p = success_error(gt_center, tracker_center, thresholds, n_frame)
                print("precision: %.4f, AUC: %.4f" % (a_p[20], np.mean(a_o)))

            if save:
                # save results
                if 'VOT2018-LT' == dataset.name:
                    video_path = os.path.join('results', dataset.name, name, 'longterm', video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x]) + '\n')
                    result_path = os.path.join(video_path,'{}_001_confidence.value'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in scores:
                            f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                    result_path = os.path.join(video_path,'{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                elif 'GOT-10k' == dataset.name:
                    video_path = os.path.join('results', dataset.name, name, video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x]) + '\n')
                    result_path = os.path.join(video_path, '{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                else:
                    model_path = os.path.join('results', dataset.name, name)
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)
                    result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x]) + '\n')
