from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import math

import torch.utils.data as data
import torch
import cv2

from lib.utils.image import get_affine_transform, affine_transform
from lib.utils.image import gaussian_radius, draw_umich_gaussian
from lib.utils.image import draw_dense_reg
from lib.utils.opts import opts
from lib.utils.augmentations import Augmentation


class COCO(data.Dataset):
    num_classes         = 1
    default_resolution  = [512, 512]
    dense_wh            = False
    reg_offset          = True
    mean = np.array([0.49965, 0.49965, 0.49965], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.08255, 0.08255, 0.08255], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(COCO, self).__init__()
        self.opt   = opt
        self.split = split

        self.img_dir0 = opt.data_dir
        self.img_dir  = os.path.join(opt.data_dir, 'images', split)

        # Resolution: 512×512 train, 1024×1024 test (when test_large_size)
        if opt.test_large_size and split != 'train':
            self.resolution = [1024, 1024]
        else:
            self.resolution = [512, 512]

        self.annot_path = os.path.join(
            opt.data_dir, 'annotations', f'instances_{split}.json'
        )

        self.down_ratio = opt.down_ratio     # 4
        self.max_objs   = opt.K              # 128
        self.seqLen     = opt.seqLen         # 5

        self.class_name = ['__background__', 'car']
        self._valid_ids = [1]
        self.cat_ids    = {v: i for i, v in enumerate(self._valid_ids)}

        print(f'==> initialising VISO {split} | res={self.resolution}')
        self.coco        = coco.COCO(self.annot_path)
        self.images      = self.coco.getImgIds()
        self.num_samples = len(self.images)
        print(f'    {self.num_samples} samples loaded')

        self.aug = Augmentation() if split == 'train' else None

    # ── helpers ────────────────────────────────────────────────────────────

    def _to_float(self, x):
        return float(f'{x:.2f}')

    def _coco_box_to_bbox(self, box):
        """COCO [x,y,w,h] → xyxy"""
        return np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]],
            dtype=np.float32
        )

    # ── evaluation ─────────────────────────────────────────────────────────

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id, cls_dict in all_bboxes.items():
            for cls_ind, bboxes in cls_dict.items():
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in bboxes:
                    b = [float(bbox[0]), float(bbox[1]),
                         float(bbox[2] - bbox[0]),    # x1y1wh for COCO
                         float(bbox[3] - bbox[1])]
                    detections.append({
                        'image_id'   : int(image_id),
                        'category_id': int(category_id),
                        'bbox'       : list(map(self._to_float, b)),
                        'score'      : float(f'{bbox[4]:.2f}'),
                    })
        return detections

    def save_results(self, results, save_dir, time_str):
        path = f'{save_dir}/results_{time_str}.json'
        json.dump(self.convert_eval_format(results), open(path, 'w'))
        print(f'Saved → {path}')

    def run_eval(self, results, save_dir, time_str):
        self.save_results(results, save_dir, time_str)
        coco_dets = self.coco.loadRes(f'{save_dir}/results_{time_str}.json')
        coco_eval = COCOeval(self.coco, coco_dets, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats, coco_eval.eval['precision']

    def __len__(self):
        return self.num_samples

    # ── __getitem__ ────────────────────────────────────────────────────────

    def __getitem__(self, index):
        # ── 1. Identify image ──────────────────────────────────────────────
        img_id    = self.images[index]
        img_info  = self.coco.loadImgs(ids=[img_id])[0]
        file_name = img_info['file_name']

        # Original dimensions stored in JSON (data_prep saved these)
        orig_w = img_info['width']
        orig_h = img_info['height']

        # ── 2. Parse video_id / frame_id ──────────────────────────────────
        base     = os.path.splitext(file_name)[0]    # "001_000123"
        parts    = base.split('_')
        video_id = parts[0]
        frame_id = int(parts[1])
        imtype   = os.path.splitext(file_name)[1]

        # ── 3. Load T=seqLen frames ────────────────────────────────────────
        seq_num  = self.seqLen
        new_h, new_w = self.resolution            # 512 or 1024

        img = np.zeros([new_h, new_w, 3, seq_num], dtype=np.float32)

        curr_path = os.path.join(self.img_dir, file_name)
        imgOri    = cv2.imread(curr_path)
        if imgOri is None:
            raise RuntimeError(f'Cannot read: {curr_path}')

        imgOri_resized = cv2.resize(imgOri, (new_w, new_h))

        for ii in range(seq_num):
            prev_frame = max(frame_id - ii, 1)
            im_name    = f'{video_id}_{prev_frame:06d}{imtype}'
            im_path    = os.path.join(self.img_dir, im_name)
            im         = cv2.imread(im_path)
            if im is None:
                im = imgOri.copy()
            im    = cv2.resize(im, (new_w, new_h))
            inp_i = (im.astype(np.float32) / 255. - self.mean) / self.std
            img[:, :, :, ii] = inp_i

        # ── 4. Load GT annotations (in original coords) ────────────────────
        ann_ids  = self.coco.getAnnIds(imgIds=[img_id])
        anns     = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        # ── 5. Coordinate setup ────────────────────────────────────────────
        # Scale factors: original → current resolution (for heatmap building)
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        # ┌─────────────────────────────────────────────────────────────────┐
        # │  THE KEY FIX                                                    │
        # │                                                                 │
        # │  c and s MUST be based on ORIGINAL image size.                 │
        # │                                                                 │
        # │  ctdet_post_process inverts the affine transform using c & s   │
        # │  to map predictions from heatmap space → original image space. │
        # │  GT annotations are stored in original space.                  │
        # │  So both end up in original space → COCO eval gives correct AP.│
        # │                                                                 │
        # │  If c=[256,256], s=512 → predictions decode to 512×512 space  │
        # │  but GT is in original space → mismatch → AP = 0              │
        # └─────────────────────────────────────────────────────────────────┘
        c = np.array([orig_w / 2., orig_h / 2.], dtype=np.float32)
        s = max(orig_w, orig_h) * 1.0

        # Affine transform for building heatmap (maps orig coords → output)
        output_h     = new_h // self.down_ratio    # 128 (train) or 256 (test)
        output_w     = new_w // self.down_ratio
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        # ── 6. Build heatmap targets ───────────────────────────────────────
        hm       = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh       = np.zeros((self.max_objs, 2),  dtype=np.float32)
        reg      = np.zeros((self.max_objs, 2),  dtype=np.float32)
        ind      = np.zeros(self.max_objs,        dtype=np.int64)
        reg_mask = np.zeros(self.max_objs,        dtype=np.uint8)

        bbox_tol   = []
        cls_id_tol = []

        for k in range(num_objs):
            ann    = anns[k]
            bbox   = self._coco_box_to_bbox(ann['bbox'])   # xyxy in ORIGINAL coords
            cls_id = self.cat_ids[ann['category_id']]
            bbox_tol.append(bbox)
            cls_id_tol.append(cls_id)

        # Augmentation (train only) — operates in original coord space
        if self.aug is not None and num_objs > 0:
            bbox_tol   = np.array(bbox_tol)
            cls_id_tol = np.array(cls_id_tol)
            img, bbox_tol, cls_id_tol = self.aug(img, bbox_tol, cls_id_tol)
            bbox_tol   = bbox_tol.tolist()
            cls_id_tol = cls_id_tol.tolist()
            num_objs   = len(bbox_tol)

        gt_det = []
        for k in range(num_objs):
            bbox   = np.array(bbox_tol[k], dtype=np.float32)
            cls_id = int(cls_id_tol[k])

            # Affine-transform from original image space → output heatmap space
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)

            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            h = np.clip(h, 0, output_h - 1)
            w = np.clip(w, 0, output_w - 1)

            if h > 0 and w > 0:
                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))))
                ct     = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    dtype=np.float32
                )
                ct[0]  = np.clip(ct[0], 0, output_w - 1)
                ct[1]  = np.clip(ct[1], 0, output_h - 1)
                ct_int = ct.astype(np.int32)

                draw_umich_gaussian(hm[cls_id], ct_int, radius)
                wh[k]       = 1. * w, 1. * h
                ind[k]      = ct_int[1] * output_w + ct_int[0]
                reg[k]      = ct - ct_int
                reg_mask[k] = 1
                gt_det.append([ct[0]-w/2, ct[1]-h/2, ct[0]+w/2, ct[1]+h/2, 1, cls_id])

        # Transpose: (H,W,3,T) → (3,T,H,W) → (3*T, H, W) [MICPL input format]
        inp = img.transpose(2, 3, 0, 1).astype(np.float32)

        # ── 7. Return ──────────────────────────────────────────────────────
        ret = {
            'input'    : inp,
            'hm'       : hm,
            'reg_mask' : reg_mask,
            'ind'      : ind,
            'wh'       : wh,
            'reg'      : reg,
            'file_name': file_name,
            'imgOri'   : imgOri_resized,
            'orig_size': np.array([orig_h, orig_w], dtype=np.float32),
            'meta': {
                'c'     : c,       # ← based on ORIGINAL size
                's'     : s,       # ← based on ORIGINAL size
                'img_id': img_id,  # ← needed by save_result in ctdet.py
            }
        }

        return img_id, ret