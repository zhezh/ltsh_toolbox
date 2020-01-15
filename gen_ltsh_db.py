import os
from pathlib import Path
import scipy
import scipy.io
import numpy as np
import pandas as pd
import re
import pickle
import skimage
import skimage.io
from tqdm import tqdm

from utils import *


def main(basepath):
    db = []
    # get all sub folder
    basepath = Path(basepath)
    assert os.path.exists(basepath), 'Check the dataset base path!'
    for imp in tqdm(basepath.glob('train/*')):
        out_annots = process_a_group(imp)
        if out_annots is None:
            continue
        kp = out_annots['annolist']  # (nimgs, npeople, njoint, 4)
        kp2d = kp[:,:,:,0:2]
        kpvis = kp[:,:,:,2]
        kpid = kp[:,:,:,3]

        img_file = out_annots['annolist_img']
        # im = skimage.io.imread(basepath/img_file[0])

        head_boxes = out_annots['annolist_head']
        objpos = out_annots['annolist_objpos']
        scale_single = out_annots['annolist_scale']
        scale = np.stack([scale_single, scale_single], axis=2)

        nimgs, npeople = kp.shape[0:2]
        for i in range(nimgs):
            for p in range(npeople):
                img_str = img_file[i]
                head_box = head_boxes[i, p]
                kp2d_cur = kp2d[i, p]
                kpvis_cur = kpvis[i, p]
                center_cur = objpos[i, p]
                scale_cur = scale[i, p]
                c, s = get_bbox(center_cur, scale_cur, kp2d_cur)

                joints_3d = np.concatenate([kp2d_cur, np.zeros((16,1))], axis=1)
                joints_3d_vis = np.stack([kpvis_cur, kpvis_cur, np.zeros(16)], axis=1)

                datum = {
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'image': img_str,
                    'center': c,
                    'scale': s,
                    'filename': '',
                    'imgnum': 0,
                    'headbox': head_box,
                }
                db.append(datum)

    return db


if __name__ == '__main__':
    basepath = Path('data/ltsh')
    db = main(basepath)
    db_path = Path('data/annot')/'train.pkl'
    with open(db_path, 'wb') as f:
        pickle.dump(db, f)
