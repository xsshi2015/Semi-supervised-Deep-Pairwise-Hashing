'''
Author: Manish Sapkota
Code similar to matlab code provided by by Rob Fergus
compact bit represetation of binary values
'''
import numpy as np
import math
from bitstring import BitArray as BA

def compactbit(b):
    b = np.array(b)
    nSamples, nbits = b.shape
    nwords = np.uint8(math.ceil(float(nbits)/8.0))
    cb = np.zeros((nSamples, nwords)).astype(np.uint8)

    for j in range(0, nbits):
        w = np.uint8(math.ceil(float(j+1)/8.0)) - 1
        cb_tmp = [BA('uint:8='+str(x)) for x in cb[:, w]]
        for x, y in zip(cb_tmp, b[:, j]):
            x.set(y, (7 - (j % 8)))

        cb_tmp = [x.uint for x in cb_tmp]
        cb[:, w] = cb_tmp

    return cb