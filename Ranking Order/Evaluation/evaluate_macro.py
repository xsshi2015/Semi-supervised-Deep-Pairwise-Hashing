'''
Author: Manish Sapkota
Code similar to matlab code provided by by Rob Fergus
'''
import numpy as np
import math

def evaluate_macro(Rel, Ret):
    # Rel is the true relevent relations
    # Ret is the predict relevent relations
    
    numtrain, numtest = Rel.shape
    precision = np.squeeze(np.zeros((1, numtest)))
    recall = np.squeeze(np.zeros((1, numtest)))


    retrieved_relevant_pairs = (Rel & Ret)

    for j in range(0, numtest):
        retrieved_relevant_num = np.count_nonzero(retrieved_relevant_pairs[:,j])
        retrieved_num = np.count_nonzero(Ret[:,j])
        relevant_num  = np.count_nonzero(Rel[:,j])
        if retrieved_num:
            precision[j] = retrieved_relevant_num / retrieved_num
        else:
            precision[j] = 0
        
        if relevant_num:
            recall[j] = retrieved_relevant_num / relevant_num
        else:
            recall[j] = 0
        
    

    p = np.mean(precision)
    r = np.mean(recall)


    return p, r
    