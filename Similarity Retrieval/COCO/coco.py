from __future__ import print_function
'''
sentence processing code modified from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
'''
from pycocotools.coco import COCO
import numpy as np
from random import shuffle
# from orderedset import OrderedSet
from collections import defaultdict
from PIL import Image
import os as os
from log_utils import user_log
import tables
import pandas as pd

import _pickle as cPickle

from pprint import pprint
# from data_helpers import *
# from w2v import train_word2vec

nrows, ncols, nch = 128, 128, 3

def load_batch(data_type,
                sequence_length=128,
                embedding_dim=128, context=10,
                min_word_count=1, nch_sent=5):
    data_dir = '.'
    annFile = '{}/data/coco/annotations/instances_{}.json'.format(data_dir, data_type)
    # capsFile = '{}/data/coco/annotations/captions_{}.json'.format(data_dir, data_type)
    hdf5_path = '{}/data/coco/dataset_{}.h5'.format(data_dir, data_type) # path to save the hdf5 file
    # voc_pick_File = '{}/data/coco/mr.p'.format(data_dir)
    #w2v_file = './data/coco/GoogleNews-vectors-negative300.bin'

    if not os.path.exists(hdf5_path):
        print ("Tere is no this file!")
        # # initialize COCO api for instance annotations
        # coco = COCO(annFile)
        # # coco_caps = COCO(capsFile)

        # cats = coco.loadCats(coco.getCatIds())
        # nms = [cat['name'] for cat in cats]
        # sup_nms = OrderedSet([cat['supercategory'] for cat in cats])

        # # get all images containing given categories
        # catIds = coco.getCatIds(catNms=nms)
        # imgIds = coco.getImgIds()
        

        # data_shape = (len(imgIds), nrows, ncols, nch)
        # fine_label_shape = (len(imgIds), 80)
        # coarse_label_shape = (len(imgIds), )

        # images = np.ndarray(shape=data_shape, dtype=np.uint8)
        # fine_labels = np.ndarray(shape=fine_label_shape, dtype=np.uint8)
        # coarse_labels = np.ndarray(shape=coarse_label_shape, dtype=np.uint8)
        # user_log("Shape till now: ", images.shape)

        # idx  = 0 #index for images
        # # sentences = []
        # # tmp_captions = []
        # # import pdb; pdb.set_trace()
        # for imgId in imgIds:
        #     label = [0] * 80
        #     for i, catId in enumerate(catIds):
        #         ids = coco.getImgIds(catIds=catId) # get images of this particular category
        #         if imgId in ids:
        #             label[i] = 1
        #             cat = coco.loadCats(catId)[0]
        #             coarse_labels[idx] = sup_nms.index(cat['supercategory'])
                    
        #     fine_labels[idx] = label
        #     img = coco.loadImgs(imgId)[0]
        #     path = ('%s/data/coco/%s/%s'%(data_dir, data_type, img['file_name']))
        #     image = Image.open(path).convert('RGB')
        #     image = image.resize((nrows, ncols), Image.ANTIALIAS)
        #     images[idx] = np.asarray(image)
        #     user_log("Shape till now: ", images[idx].shape)
        #     idx += 1
        # #     # load and display caption annotations
        # #     annIds = coco_caps.getAnnIds(imgIds=img['id'])
        # #     anns = coco_caps.loadAnns(annIds)
        # #     tmp_caption = []
        # #     for i, ann in enumerate(anns):
        # #         sentences.append(ann)
        # #         user_log("Loading caption " +str(i) +" for image: "+str(idx))
        # #         if i > nch_sent-1:
        # #             continue
        # #         tmp_caption.append(ann)

        # #     tmp_captions.append(tmp_caption)
        

        # # if not os.path.exists(voc_pick_File):
        # #     user_log("loading data...")
        # #     """
        # #     Loads and preprocessed data for the MR dataset.
        # #     Returns input vectors, labels, vocabulary, and inverse vocabulary.
        # #     """
        # #     # Load and preprocess data
        # #     sentences_padded, sequence_length = pad_sentences(sentences, max_length=sequence_length)
        # #     vocab, vocab_inv = build_vocab(sentences_padded)
        # #     data_x = build_input_data(sentences_padded, vocab)
        # #     vocab_inv = {key: value for key, value in enumerate(vocab_inv)}

        # #     user_log("data loaded!")
        # #     user_log("number of sentences: " + str(len(data_x)))
        # #     user_log("vocab size: " + str(len(vocab)))
        # #     user_log("max sentence length: " + str(sequence_length))
        # #     user_log("loading word2vec vectors...")
        # #     embedding_weights = train_word2vec(data_x,
        # #                         vocab_inv,
        # #                         num_features=embedding_dim,
        # #                         min_word_count=min_word_count,
        # #                         context=context)
        # #     cPickle.dump([vocab, vocab_inv, sequence_length, embedding_weights], open(voc_pick_File, "wb"))
        # #     user_log("dataset created!")
        # # else:
        # #     x = cPickle.load(open(voc_pick_File, "rb"))
        # #     vocab, vocab_inv, sequence_length, embedding_weights = x[0], x[1], x[2], x[3]
        # #     user_log("data loaded!")

        # # caption_shape = (len(imgIds), nch_sent, sequence_length)
        # # captions = np.ndarray(shape=caption_shape, dtype=np.int)
        # # for i, tmp in enumerate(tmp_captions):
        # #     tmp_pad, _ = pad_sentences(tmp, max_length=sequence_length)
        # #     tmp_pad = build_input_data(tmp_pad, vocab) # map word to index
        # #     captions[i] = tmp_pad
        # #     user_log("shape of captions: ", captions[i].shape)

        # user_log("Saving the dataset...")
        # # open a hdf5 file and create earrays
        # with tables.open_file(hdf5_path, mode='w') as hdf5_file:
        #     # create the label arrays and copy the labels data in them
        #     hdf5_file.create_array(hdf5_file.root, 'images', images)
        #     hdf5_file.create_array(hdf5_file.root, 'fine_labels', fine_labels)
        #     hdf5_file.create_array(hdf5_file.root, 'coarse_labels', coarse_labels)
            # hdf5_file.create_array(hdf5_file.root, 'captions', captions)
    else:
        # x = cPickle.load(open(voc_pick_File, "rb"))
        # _, vocab_inv, _, embedding_weights = x[0], x[1], x[2], x[3]
        # user_log("data loaded!")

        user_log("Loading the dataset...")
        with tables.open_file(hdf5_path, mode='r') as hdf5_file:
            images = hdf5_file.root.images[:]
            fine_labels = hdf5_file.root.fine_labels[:]
            coarse_labels = hdf5_file.root.coarse_labels[:]
            # captions = hdf5_file.root.captions[:]

    return images, fine_labels, coarse_labels #, captions, embedding_weights, len(vocab_inv)

def generate_ranking_info(batch_label):
    '''
    (fine_labels, coarse_labels) = y_train[i][0], y_train[i][1]
    '''
    n = batch_label.shape[0]
    S = -np.ones((n, n))
    for iter in range(n):
        for jter in range(n):
            tepV = np.matmul(batch_label[iter, :], np.transpose(batch_label[jter, :]))
            if tepV != 0:
                S[iter, jter] = tepV
    return S

def load_data(data_type):
    if data_type == 'train':
        images, fine_labels, coarse_labels = load_batch('train2014')
    else:
        images, fine_labels, coarse_labels = load_batch('val2014')

    return images, fine_labels, coarse_labels