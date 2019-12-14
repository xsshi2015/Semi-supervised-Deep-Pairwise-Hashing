'''
Author: Manish Sapkota
Code similar to matlab code provided by by Rob Fergus
'''
import numpy as np
import math

def cat_apcal(traingnd, testgnd, IX, rn):
    #  ap=apcal(score,label)
    # % average precision (AP) calculation 
    S = np.matmul(testgnd, np.transpose(traingnd))
    S[S>0]=1


    numtrain, numtest = IX.shape
    apall = np.zeros((1, numtest))
    for i in range(0, numtest):
        y = IX[:, i]
        x = 0
        p = 0
        new_label = S[i,:]
        
        num_return_NN = rn
        for j in range(0, num_return_NN):
            if new_label[y[j]] == 1:
                x = x+1
                p = p+x/float((j+1))
        
        if p == 0:
            apall[:, i] = 0
        else:
            apall[:, i] = p/float(x)

    return np.mean(apall, axis=1)