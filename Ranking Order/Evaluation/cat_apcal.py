'''
Author: Manish Sapkota
Code similar to matlab code provided by by Rob Fergus
'''
import numpy as np
import math

def cat_apcal(traingnd, testgnd, IX, num=500):
    #  ap=apcal(score,label)
    # % average precision (AP) calculation 
    
    numtrain, numtest = IX.shape
    apall = np.zeros((1, numtest))
    for i in range(0, numtest):
        y = IX[:, i]
        x = 0
        p = 0
        new_label = np.zeros((1,numtrain))
        inx = np.squeeze(np.where(traingnd==testgnd[i]))
        new_label[:, inx] = 1
        
        num_return_NN = min(5000, traingnd.shape[0])
        
        for j in range(0, num_return_NN):
            if new_label[:, y[j]] == 1:
                x = x+1
                p = p+x/float((j+1))
        
        if p == 0:
            apall[:, i] = 0
        else:
            apall[:, i] = p/float(x)

    return np.mean(apall, axis=1)