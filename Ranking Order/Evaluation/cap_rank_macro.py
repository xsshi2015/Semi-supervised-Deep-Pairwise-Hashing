'''
Author: Manish Sapkota
Code similar to matlab code provided by by Rob Fergus
'''
import numpy as np
import math

def cap_rank_macro(Rel, Ret):
    #  Rel is true relevant relations
    #  Ret is the predict sort index
    
    numtrain, numtest = Rel.shape
    
    rel = np.squeeze(np.zeros((1, numtest)))
    ndcg = np.squeeze(np.zeros((1, numtest)))
    z = np.zeros((1, numtest))
    sortedRel = np.sort(Rel, axis=0)

    for iter in range(0, numtest):
        idx = np.squeeze(np.where(Ret[:,iter]))
        if idx.size>0:
            rel[iter] =1/len(idx)*np.sum(Rel[idx,iter])

            for jter in range(0, idx.size):
                ndcg[iter] = ndcg[iter] +(2**Rel[iter,Ret[idx[jter], iter]]-1)/np.log2(jter+2)
                z[iter] = z[iter] + (2**sortedRel[jter, iter]-1)/np.log2(jter+2)

        if z[iter]==0:
            ndcg[iter]=0
        else:
            ndcg[iter] = ndcg[iter]/z[iter]
        
    r = np.mean(rel)
    n = np.mean(ndcg)

    return r, n
            