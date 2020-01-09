import os
from pathlib import Path
import scipy
import scipy.io
import numpy as np
import pandas as pd
import re


def _process_one_cell(cell):
    mdtype = cell.dtype
    if mdtype.names is not None:
        ndata = [np.array(cell[n][0, 0])[0, 0] for n in mdtype.names]
    else:
        ndata = np.ones(4) * -1
    return ndata


def process_a_group(basepath, regex_pattern):
    """

    :param basepath: the path e.g. 'ltsh/train/145'
    :return:
    """
    basepath = Path(basepath)
    anno_keys = {'annolist': 'annolist',  # filename: keyword in mat; joint position, (vis, y, x, idx)
                 'annolist_head': 'annolist_head',  # head bbox
                 'annolist_img': 'annolist_img',  # img file path
                 'annolist_objpos': 'annolist_objpos',  # person position, one point
                 'annolist_poseExtremeness': 'poseExtremeness',
                 'annolist_scale': 'annolist_scale',
                 }
    anno_paths = {k: basepath/'{}.mat'.format(k) for k in anno_keys}

    annotation = dict()
    for k in anno_keys:
        anno_p = anno_paths[k]
        anno_k = anno_keys[k]
        m = scipy.io.loadmat(anno_p)
        mdata = m[anno_k]

        if k in ['annolist']:
            # mdata 51, 9, ?
            mdtype = mdata[0, 0, 0].dtype
            ndata_shape = list(mdata.shape)
            ndata_shape.append(4)
            ndata = np.zeros(ndata_shape)
            for i in range(ndata_shape[0]):
                for ii in range(ndata_shape[1]):
                    for iii in range(ndata_shape[2]):
                        cell_data = _process_one_cell(mdata[i,ii,iii])
                        ndata[i,ii,iii] = cell_data
            annotation[k] = ndata

        elif k in ['annolist_img']:
            ndata = list(mdata)
            ndata_shorten = []
            for s in ndata:
                ss = regex_pattern.search(str(s))[0]
                ndata_shorten.append(ss)
            assert len(ndata_shorten) == len(ndata)
            annotation[k] = ndata_shorten
        else:
            ndata = np.array(mdata)
            annotation[k] = ndata

    return annotation


if __name__ == '__main__':
    base_path = Path('/data/zhaz/dataset/ltsh/train/145/')
    re_img_name = re.compile('train/[0-9]*/[A-Za-z0-9/.]*', flags=0)
    re_img_name = re.compile(re_img_name)
    out = process_a_group(base_path, re_img_name)

    # load img test
    # 51 pictures  9 human
    import skimage
    import skimage.io
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    # img_p = base_path / out['annolist_img'][1]
    img_p = base_path / 'composition/00001.png'
    img = skimage.io.imread(img_p)
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()

    # plt head box
    head_boxes = out['annolist_head'][1]
    for hb in head_boxes:
        lt = np.array([hb[0], hb[1]])
        size = np.array([hb[2]-hb[0], hb[3]-hb[1]])
        p = patches.Rectangle(lt, size[0], size[1], fill=False)
        ax.add_patch(p)

    objpos = out['annolist_objpos'][1]
    for ob in objpos:
        ax.plot(ob[0], ob[1], color='red', marker='o', linewidth=2, markersize=12)

    kp = out['annolist'][1]  # (9, 16, 4)
    # ('is_visible', 'O'), ('y', 'O'), ('x', 'O'), ('id', 'O') idx of joint [0, 15]
    for kpp in kp:
        for j in kpp:
            vis = j[0]
            y = j[1]
            x = j[2]
            idx_joint = j[3]
            ax.plot(x, y, color='gold', marker='+', linewidth=2, markersize=6)



    pass
