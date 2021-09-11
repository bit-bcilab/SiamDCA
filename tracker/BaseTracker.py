

import numpy as np
import cv2


class BaseSiamTracker(object):
    def __init__(self, model, model_cfg, session, weight_path, mode):
        pass

    def init(self, img, bbox, video_name):
        pass

    def track(self, img, cfg):
        pass


def change(r):
    return np.maximum(r, 1. / r)


def sz(w, h, context_amount=0.5):
    pad = (w + h) * context_amount
    return np.sqrt((w + pad) * (h + pad))


def get_heatmaps(x_crop, cls_, num_anchors, wr=0.4, wi=0.6):
    size = x_crop.shape[1]
    for i in range(num_anchors):
        response = cv2.resize(cls_[..., i], (size, size))
        response = np.uint8(response * 255.)
        response = cv2.applyColorMap(response, cv2.COLORMAP_JET)
        superimposed_img = response * wr + x_crop[0, ...] * wi
        superimposed_img = np.uint8(superimposed_img)
        if i == 0:
            heat_map = superimposed_img
        else:
            heat_map = np.concatenate([heat_map, superimposed_img], axis=1)
    return heat_map


def grad_cam(f, grad):
    grad_ = grad.mean((0, 1))
    f_ = f * grad_
    cam = f_.mean(-1)
    cam_ = cam / cam.max()
    return cam_
