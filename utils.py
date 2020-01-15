import os
from pathlib import Path
import scipy
import scipy.io
import numpy as np
import pandas as pd
import re


__re_img_name = None


def _process_kp2d_cell(cell):
    """
    convert matlab cell with 2d joint annotations to np array with fixed attributes order
    :param cell: a matlab cell
    :return:
    """
    mdtype = cell.dtype
    if mdtype.names is not None:
        # ndata = [np.array(cell[n][0, 0])[0, 0] for n in mdtype.names]
        ndata_ = {n: np.array(cell[n][0, 0])[0, 0] for n in mdtype.names}
        ndata = [ndata_[k] for k in ['x','y','is_visible', 'id'] ]
    else:
        ndata = np.ones(4) * -1
    return ndata


def process_a_group(basepath):
    """

    :param basepath: the path e.g. 'ltsh/train/145'
    :return: all the related annotations in a dict
    """
    global __re_img_name
    if __re_img_name is None:
        __re_img_name = re.compile('train/[0-9]*/[A-Za-z0-9/.]*', flags=0)

    basepath = Path(basepath)
    assert os.path.exists(basepath), 'No such directory {}'.format(basepath)

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
        try:
            m = scipy.io.loadmat(anno_p)
        except OSError:
            print('file not exist {}'.format(anno_p))
            return None
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
                        cell_data = _process_kp2d_cell(mdata[i, ii, iii])
                        ndata[i,ii,iii] = cell_data
            annotation[k] = ndata

        elif k in ['annolist_img']:
            ndata = list(mdata)
            ndata_shorten = []
            for s in ndata:
                ss = __re_img_name.search(str(s))[0]
                ndata_shorten.append(ss)
            assert len(ndata_shorten) == len(ndata)
            annotation[k] = ndata_shorten
        else:
            ndata = np.array(mdata)
            annotation[k] = ndata

    return annotation


def get_bbox(center, scale, kps):
    """
    process one image at a time
    :param center:
    :param scale: s * 200 is the box size in pixels
    :param kps:
    :return: bbox in (center, scale) format
    """
    # maybe return a tight bounding box also
    # The scale keeps the height of the person as about 200 px
    s = scale[0] * 200
    offset_pxs = 20/165 * s  # in pixel
    tight_tl = np.zeros(2)
    tight_br = np.zeros(2)
    kps = kps[np.logical_and(kps[:,0]>-1, kps[:,1]>-1)]
    if kps.shape[0] < 1:
        tight_tl = [1e6, 1e6]
        tight_br = [-1e6, -1e6]
    else:
        tight_tl[0] = np.min(kps[:, 0]) - offset_pxs
        tight_tl[1] = np.min(kps[:, 1]) - offset_pxs
        tight_br[0] = np.max(kps[:, 0]) + offset_pxs
        tight_br[1] = np.max(kps[:, 1]) + offset_pxs

    # compute the box in mpii way
    c = center.copy()
    if c[0] != -1:
        c[1] = c[1] + 25 * scale[1]
        s = scale * 1.25

    # MPII uses matlab format, index is based 1,
    # we should first convert to 0-based index
    c = c - 1
    mpii_tl = c - s*100
    mpii_br = c + s*100

    tl_x, tl_y = min(tight_tl[0], mpii_tl[0]), min(tight_tl[1], mpii_tl[1])
    br_x, br_y = max(tight_br[0], mpii_br[0]), max(tight_br[1], mpii_br[1])

    c_final = np.array([(tl_x+br_x)/2., (tl_y+br_y)/2.])
    s_final = max(br_y-tl_y, br_x-tl_x)
    s_final = s_final/200.
    s_final = np.array([s_final, s_final])

    return c_final, s_final


if __name__ == '__main__':
    base_path = Path('./data/ltsh/train/145/')
    base_path = Path('./data/ltsh/train/118/')
    imgi = 45
    # re_img_name = re.compile('train/[0-9]*/[A-Za-z0-9/.]*', flags=0)
    # re_img_name = re.compile(re_img_name)
    out = process_a_group(base_path)

    # load img test
    # 51 pictures  9 human
    import skimage
    import skimage.io
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.ion()
    # img_p = base_path / out['annolist_img'][1]
    img_p = base_path / 'composition/{:05d}.png'.format(imgi)
    img = skimage.io.imread(img_p)
    fig, ax = plt.subplots()
    ax.imshow(img)
    # plt.show()

    # plt head box
    head_boxes = out['annolist_head'][imgi]
    for hb in head_boxes:
        lt = np.array([hb[0], hb[1]])
        size = np.array([hb[2]-hb[0], hb[3]-hb[1]])
        p = patches.Rectangle(lt, size[0], size[1], fill=False)
        ax.add_patch(p)

    kp = out['annolist'][imgi]  # (9, 16, 4)
    kp2d = kp[:,:, 0:2]
    root = kp[:,6, 0:2]
    # root = np.zeros_like(root_yx)
    # root[:,0] = root_yx[:,1]
    # root[:, 1] = root_yx[:, 0]
    # ('is_visible', 'O'), ('y', 'O'), ('x', 'O'), ('id', 'O') idx of joint [0, 15]
    for kpp in kp:
        for j in kpp:
            vis = j[2]
            y = j[1]
            x = j[0]
            idx_joint = j[3]
            ax.plot(x, y, color='g', marker='*', linewidth=2, markersize=6)

    objpos = out['annolist_objpos'][imgi]
    scale = out['annolist_scale'][imgi]
    scale = np.array([scale, scale]).reshape((2,-1)).T
    # for ob, r, s in zip(objpos, root, scale):
    #     # ax.plot(ob[0], ob[1], color='red', marker='o', linewidth=2, markersize=12)
    #
    #     # use root as center, what the root coordinates if root is outside img ???
    #     # ANS: root joint will be (-1, -1),
    #     # but objpos is the actual negative number reflecting object position
    #
    #     s = np.array([s, s]).reshape((-1,))
    #     s = s * 1.25
    #     if r[0] > -1 and r[1] > -1:
    #         c = r
    #     else:
    #         c = np.array(ob)
    #         if c[0] != -1:
    #             c[1] = c[1] + 40 * s[1]
    #         c = c - 1
    #
    #     box_lt = c - s * 100
    #     box_size = s * 200
    #     p = patches.Rectangle(box_lt, box_size[0], box_size[1], fill=False, color='gold')
    #     ax.add_patch(p)
    boxes = []
    for i in range(len(objpos)):
        box_ = get_bbox(objpos[i], scale[i], kp2d[i])
        boxes.append(box_)

    for box in boxes:
        c, s = box[0], box[1]
        box_lt = c - s * 100
        box_size = s * 200
        p = patches.Rectangle(box_lt, box_size[0], box_size[1], fill=False, color='gold')
        ax.add_patch(p)


    extre = out['annolist_poseExtremeness'][imgi]

    pass
