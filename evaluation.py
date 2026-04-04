import numpy as np
import scipy.io as sio
import os
import xml.etree.ElementTree as ET
from lib.utils.utils_eval import eval_metric

if __name__ == '__main__':
    eval_mode_metric = 'iou'
    conf_thresh = 0.3
    
    # 🔥 The exact videos the authors evaluated on
    TARGET_VIDEOS = [3, 5, 2, 8, 10, 6, 9]
    
    ANN_PATH = '/kaggle/input/datasets/shub7ham/viso-sod/VISO/Detection_coco_format/coco/car/Annotations/test2017/'
    results_dir0 = '/kaggle/working/exp/results/VISO/MICPL/results/MICPL_models_model_last_bifpn_mat/' 

    print(f"Reading predictions from: {results_dir0}")
    
    dis_th_cur = 5
    iou_th_cur = 0.0
    
    det_metric = eval_metric(dis_th=dis_th_cur, iou_th=iou_th_cur, eval_mode=eval_mode_metric)
    det_metric.reset()
    
    anno_files = [f for f in os.listdir(ANN_PATH) if f.endswith('.xml')]
    num_images = len(anno_files)
    
    processed_count = 0
    
    for index in range(num_images):
        file_name = anno_files[index]
        annName = os.path.join(ANN_PATH, file_name)
        
        # 🔥 Dynamic Video Filtering: Read the XML to check the video ID
        tree = ET.parse(annName)
        folder_text = tree.getroot().find('folder').text
        video_id = int(folder_text.replace('\\', '/').split('/')[-2])
        
        if video_id not in TARGET_VIDEOS:
            continue # Skip this XML, it's not in our target list!
            
        processed_count += 1
        
        gt_t = det_metric.getGtFromXml(annName)
        
        # Load the corresponding .mat prediction
        matname = os.path.join(results_dir0, file_name.replace('.xml','.mat'))
        
        if os.path.exists(matname):
            det_ori = sio.loadmat(matname)['A']
            if len(det_ori) > 0:
                det = np.array(det_ori)
                score = det[:,-1]
                inds = np.argsort(-score)
                score = score[inds]
                det = det[score > conf_thresh]
            else:
                det = np.empty([0,4])
        else:
            det = np.empty([0,4])
            
        det_metric.update(gt_t, det)
        
    result = det_metric.get_result()
    print(f"\n✅ Evaluated {processed_count} images from target videos.")
    print('evalmode=%s, conf_th=%0.2f, re=%0.3f, prec=%0.3f, f1=%0.3f' % (
        eval_mode_metric, conf_thresh, result['recall'], result['prec'], result['f1']))