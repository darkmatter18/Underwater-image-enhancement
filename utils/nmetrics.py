import math

import numpy as np
from skimage import color, filters


def nmetrics(a):
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    _lab = lab[:, :, 0]

    # 1st term
    chroma = (lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2) ** 0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc) ** 2)) ** 0.5

    # 2nd term
    top = np.int(np.round(0.01 * _lab.shape[0] * _lab.shape[1]))
    sl = np.sort(_lab, axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[::top]) - np.mean(sl[::top])

    # 3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = _lab.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0:
            satur.append(0)
        elif l1[i] == 0:
            satur.append(0)
        else:
            satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753

    # 1st term UICM
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]
    rgl = np.sort(rg, axis=None)
    ybl = np.sort(yb, axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int(al1 * len(rgl))
    T2 = np.int(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr - uyb) ** 2)

    uicm = -0.0268 * np.sqrt(urg ** 2 + uyb ** 2) + 0.1586 * np.sqrt(s2rg + s2yb)

    # 2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:, :, 0] * filters.sobel(rgb[:, :, 0])
    Gsobel = rgb[:, :, 1] * filters.sobel(rgb[:, :, 1])
    Bsobel = rgb[:, :, 2] * filters.sobel(rgb[:, :, 2])

    Rsobel = np.round(Rsobel).astype(np.uint8)
    Gsobel = np.round(Gsobel).astype(np.uint8)
    Bsobel = np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    # 3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm, uciqe


def eme(ch, block_size=8):
    num_x = math.ceil(ch.shape[0] / block_size)
    num_y = math.ceil(ch.shape[1] / block_size)

    _eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * block_size
        if i < num_x - 1:
            xrb = (i + 1) * block_size
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * block_size
            if j < num_y - 1:
                yrb = (j + 1) * block_size
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]

            block_min = np.float(np.min(block))
            block_max = np.float(np.max(block))

            # # old version
            # if block_min == 0.0: eme += 0
            # elif block_max == 0.0: eme += 0
            # else: eme += w * math.log(block_max / block_min)

            # new version
            if block_min == 0:
                block_min += 1
            if block_max == 0:
                block_max += 1
            _eme += w * math.log(block_max / block_min)
    return _eme


def logamee(ch, block_size=8):
    num_x = math.ceil(ch.shape[0] / block_size)
    num_y = math.ceil(ch.shape[1] / block_size)

    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * block_size
        if i < num_x - 1:
            xrb = (i + 1) * block_size
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * block_size
            if j < num_y - 1:
                yrb = (j + 1) * block_size
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]
            block_min = np.float(np.min(block))
            block_max = np.float(np.max(block))

            top = plipsub(block_max, block_min)
            bottom = plipsum(block_max, block_min)

            m = top / bottom
            if m == 0.:
                s += 0
            else:
                s += m * np.log(m)

    return plipmult(w, s)


def plipsub(i, j, k=1026):
    return k * (i - j) / (k - j)


def plipmult(c, j, gamma=1026):
    return gamma - gamma * (1 - j / gamma) ** c


def plipsum(i, j, gamma=1026):
    return i + j - i * j / gamma
