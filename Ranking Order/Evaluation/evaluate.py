'''
Author: Manish Sapkota
Code similar to matlab code provided by by Rob Fergus
'''
import numpy as np
import math

def evaluate(traingnd, testgnd, IX, num_return_NN, recall_flag=0, class_num=10):
    #  ap=apcal(score,label)
    # % average precision (AP) calculation 
    
    numtrain, numtest = IX.shape
    precision = np.zeros((len(num_return_NN), numtest))
    recall = np.zeros((len(num_return_NN), numtest))
    for i in range(0, numtest):
        y = IX[:, i]
        new_label = np.zeros((1,numtrain))
        inx = np.squeeze(np.where(traingnd==testgnd[i]))
        new_label[:, inx] = 1
        
        for k in range(0, len(num_return_NN)):
            x = 0
            for j in range(0, num_return_NN[k]):
                if new_label[:, y[j]] == 1:
                    x = x+1
                    
            precision[k,i] = x/num_return_NN[k]
            if recall_flag==1:
                recall[k,i] = class_num*x/numtrain 
            else:
                recall=None

    if recall is not None:
        return  np.mean(precision, axis=1), np.mean(recall, axis=1)
    else:
        return  np.mean(precision, axis=1), recall
    