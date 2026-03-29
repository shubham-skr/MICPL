# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import os
# import numpy as np
# import time
# import torch

# from lib.utils.opts import opts
# from lib.models.stNet import get_det_net, load_model
# from lib.dataset.coco_rsdata import COCO  

# from lib.external.nms import soft_nms
# from lib.utils.decode import ctdet_decode
# from lib.utils.post_process import ctdet_post_process

# from progress.bar import Bar


# # =========================
# # MODEL FORWARD
# # =========================
# def process(model, image):
#     with torch.no_grad():
#         output = model(image)[-1]
#         hm = output['hm'].sigmoid_()
#         wh = output['wh']
#         reg = output['reg']
#         dets = ctdet_decode(hm, wh, reg=reg)
#     return output, dets


# # =========================
# # POST PROCESS
# # =========================
# def post_process(dets, meta, num_classes=1):
#     dets = dets.detach().cpu().numpy()
#     dets = dets.reshape(1, -1, dets.shape[2])

#     dets = ctdet_post_process(
#         dets.copy(), [meta['c']], [meta['s']],
#         meta['out_height'], meta['out_width'], num_classes)

#     for j in range(1, num_classes + 1):
#         dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)

#     return dets[0]


# def pre_process(image):
#     height, width = image.shape[2:4]

#     c = np.array([width / 2., height / 2.], dtype=np.float32)
#     s = max(height, width) * 1.0

#     meta = {
#         'c': c,
#         's': s,
#         'out_height': height,
#         'out_width': width
#     }

#     meta_batch = batch['meta']

#     c = meta_batch['c'][0].cpu().numpy()
#     s = meta_batch['s'][0].cpu().numpy()

#     inp_h, inp_w = batch['input'].shape[2:4]

#     meta = {
#         'c': c,
#         's': s,
#         'out_height': inp_h,
#         'out_width': inp_w
#     }
#     return meta


# def merge_outputs(detections, num_classes, max_per_image):
#     results = {}

#     for j in range(1, num_classes + 1):
#         results[j] = np.concatenate(
#             [detection[j] for detection in detections], axis=0
#         ).astype(np.float32)

#         soft_nms(results[j], Nt=0.5, method=2)

#     scores = np.hstack(
#         [results[j][:, 4] for j in range(1, num_classes + 1)]
#     )

#     if len(scores) > max_per_image:
#         thresh = np.partition(scores, len(scores) - max_per_image)[-max_per_image]
#         for j in range(1, num_classes + 1):
#             results[j] = results[j][results[j][:, 4] >= thresh]

#     return results


# # =========================
# # TEST LOOP
# # =========================
# def test(opt, split, model_path):

#     os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
#     device = torch.device('cuda')

#     print(f"\n🚀 Loading dataset: {split}")
#     dataset = COCO(opt, split)

#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=1
#     )

#     print("🔧 Building model...")
#     model = get_det_net(
#         {'hm': dataset.num_classes, 'wh': 2, 'reg': 2},
#         opt.model_name
#     )

#     model = load_model(model, model_path)
#     model = model.to(device)
#     model.eval()

#     results = {}

#     bar = Bar('Testing', max=len(loader))

#     for ind, (img_id, batch) in enumerate(loader):

#         image = batch['input'].to(device)

#         meta = pre_process(batch['input'])

#         output, dets = process(model, image)

#         dets = post_process(dets, meta, dataset.num_classes)

#         # NEW: RESCALE BACK TO ORIGINAL SIZE
#         orig_h, orig_w = batch['orig_size'][0].cpu().numpy()

#         inp_h, inp_w = batch['input'].shape[2:4]

#         scale_x = orig_w / inp_w
#         scale_y = orig_h / inp_h

#         for j in dets:
#             if len(dets[j]) > 0:
#                 dets[j][:, 0] *= scale_x
#                 dets[j][:, 2] *= scale_x
#                 dets[j][:, 1] *= scale_y
#                 dets[j][:, 3] *= scale_y

#         results_per_img = merge_outputs([dets], dataset.num_classes, opt.K)

#         results[int(img_id.numpy()[0])] = results_per_img

#         bar.next()

#     bar.finish()

#     print("\n📊 Running COCO Evaluation...")

#     stats, _ = dataset.run_eval(results, opt.save_results_dir, "results")

#     print("\n========== FINAL METRICS ==========")
#     print(f"AP     : {stats[0]:.4f}")
#     print(f"AP50   : {stats[1]:.4f}")
#     print(f"AP75   : {stats[2]:.4f}")
#     print(f"Recall : {stats[8]:.4f}")

#     print("\n💡 Approx:")
#     print(f"Precision ≈ AP50 = {stats[1]:.4f}")
#     print(f"Recall    = {stats[8]:.4f}")
#     print("===================================")


# # =========================
# # MAIN
# # =========================
# if __name__ == '__main__':

#     opt = opts().parse()

#     split = 'test'

#     if not os.path.exists(opt.save_results_dir):
#         os.makedirs(opt.save_results_dir)

#     model_path = opt.load_model

#     print(f"\n📦 Model: {model_path}")

#     test(opt, split, model_path)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch

from lib.utils.opts import opts
from lib.models.stNet import get_det_net, load_model
from lib.dataset.coco_rsdata import COCO  

from lib.external.nms import soft_nms
from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process

from progress.bar import Bar


# =========================
# MODEL FORWARD
# =========================
def process(model, image):
    with torch.no_grad():
        output = model(image)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        dets = ctdet_decode(hm, wh, reg=reg)
    return output, dets


# =========================
# POST PROCESS
# =========================
def post_process(dets, meta, num_classes=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])

    dets = ctdet_post_process(
        dets.copy(),
        [meta['c']],
        [meta['s']],
        meta['out_height'],
        meta['out_width'],
        num_classes
    )

    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)

    return dets[0]


# =========================
# MERGE OUTPUTS
# =========================
def merge_outputs(detections, num_classes, max_per_image):
    results = {}

    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0
        ).astype(np.float32)

        soft_nms(results[j], Nt=0.5, method=2)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, num_classes + 1)]
    )

    if len(scores) > max_per_image:
        thresh = np.partition(scores, len(scores) - max_per_image)[-max_per_image]
        for j in range(1, num_classes + 1):
            results[j] = results[j][results[j][:, 4] >= thresh]

    return results


# =========================
# TEST LOOP
# =========================
def test(opt, split, model_path):

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    device = torch.device('cuda')

    print(f"\n🚀 Loading dataset: {split}")
    dataset = COCO(opt, split)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    print("🔧 Building model...")
    model = get_det_net(
        {'hm': dataset.num_classes, 'wh': 2, 'reg': 2},
        opt.model_name
    )

    model = load_model(model, model_path)
    model = model.to(device)
    model.eval()

    results = {}

    bar = Bar('Testing', max=len(loader))

    for ind, (img_id, batch) in enumerate(loader):

        image = batch['input'].to(device)

        # ✅ CRITICAL: use dataset meta (original coordinate system)
        meta_batch = batch['meta']

        c = meta_batch['c'][0].cpu().numpy()
        s = meta_batch['s'][0].cpu().numpy()

        inp_h, inp_w = batch['input'].shape[2:4]

        meta = {
            'c': c,
            's': s,
            'out_height': inp_h,
            'out_width': inp_w
        }

        output, dets = process(model, image)

        dets = post_process(dets, meta, dataset.num_classes)

        results_per_img = merge_outputs(
            [dets], dataset.num_classes, opt.K
        )

        results[int(img_id.numpy()[0])] = results_per_img

        bar.next()

    bar.finish()

    print("\n📊 Running COCO Evaluation...")

    stats, _ = dataset.run_eval(results, opt.save_results_dir, "results")

    print("\n========== FINAL METRICS ==========")
    print(f"AP     : {stats[0]:.4f}")
    print(f"AP50   : {stats[1]:.4f}")
    print(f"AP75   : {stats[2]:.4f}")
    print(f"Recall : {stats[8]:.4f}")
    print("===================================")


# =========================
# MAIN
# =========================
if __name__ == '__main__':

    opt = opts().parse()

    split = 'test'

    if not os.path.exists(opt.save_results_dir):
        os.makedirs(opt.save_results_dir)

    model_path = opt.load_model

    print(f"\n📦 Model: {model_path}")

    test(opt, split, model_path)