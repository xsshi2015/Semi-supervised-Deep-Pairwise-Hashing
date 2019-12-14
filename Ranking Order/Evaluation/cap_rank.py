'''
Author: Manish Sapkota
Code similar to matlab code provided by by Rob Fergus
'''
import numpy as np
import math

def cap_rank(Rel, Ret, num_return_NN):
    #  Rel is true relevant relations
    #  Ret is the predict sort index
    
    numtrain, numtest = Rel.shape
    
    rel = np.squeeze(np.zeros((1, numtest)))
    ndcg = np.squeeze(np.zeros((1, numtest)))
    z = np.squeeze(np.zeros((1, numtest)))
    sortedRel = Rel.copy()

    sortedRel[::-1].sort(axis=0)
    
    for iter in range(0, numtest):
        rel[iter] =1/num_return_NN*np.sum(Rel[Ret[1:num_return_NN, iter], iter])
        for jter in range(0, num_return_NN):
            ndcg[iter] = ndcg[iter] +(2**Rel[Ret[jter, iter], iter]-1)/np.log2(jter+2)
            z[iter] = z[iter] + (2**sortedRel[jter, iter]-1)/np.log2(jter+2)

        if z[iter]==0:
            ndcg[iter]=0
        else:
            ndcg[iter] = ndcg[iter]/z[iter]
        
    r = np.mean(rel)
    n = np.mean(ndcg)

    return r, n
            