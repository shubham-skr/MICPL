import numpy as np
import scipy.io as sio
import os
from lib.utils.utils_eval import eval_metric

if __name__ == '__main__':
    eval_mode_metric = 'iou'
    conf_thresh = 0.3
    
    # 🔥 FIX 1: Point directly to your flat XML directory
    ANN_PATH = '/kaggle/input/datasets/shub7ham/viso-sod/VISO/Detection_coco_format/coco/car/Annotations/test2017/'
    
    # 🔥 FIX 2: Point to where testSaveMat.py just dumped the .mat files
    # (Check the exact folder name generated in your Kaggle output!)
    results_dir0 = '/kaggle/working/exp/results/MICPL_model_pre-trained_mat/' 

    print(f"Reading predictions from: {results_dir0}")
    txt_name = 'results_%s_%.2f.txt'%(eval_mode_metric, conf_thresh)
    fid = open(os.path.join(results_dir0, txt_name), 'w+')
    fid.write(results_dir0 + '(recall,precision,F1)\n')
    fid.write(eval_mode_metric + '\n')
    
    dis_th_cur = 5
    iou_th_cur = 0.0
    
    det_metric = eval_metric(dis_th=dis_th_cur, iou_th=iou_th_cur, eval_mode=eval_mode_metric)
    
    fid.write('conf_thresh=%.2f,thresh=%.2f\n'%(conf_thresh, iou_th_cur))
    print('conf_thresh=%.2f,thresh=%.2f'%(conf_thresh, iou_th_cur))
    
    det_metric.reset()
    
    # 🔥 FIX 3: Iterate directly over the flat directory
    anno_files = [f for f in os.listdir(ANN_PATH) if f.endswith('.xml')]
    num_images = len(anno_files)
    
    for index in range(num_images):
        file_name = anno_files[index]
        annName = os.path.join(ANN_PATH, file_name)
        
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
    fid.write('&%.2f\t&%.2f\t&%.2f\n' % (result['recall'], result['prec'], result['f1']))
    print('evalmode=%s, conf_th=%0.2f, re=%0.3f, prec=%0.3f, f1=%0.3f' % (
        eval_mode_metric, conf_thresh, result['recall'], result['prec'], result['f1']))
    
    fid.close()