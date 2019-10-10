# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:28:01 2019

@author: Sumangala
"""
# Unicode character encoding for each word

# This module defines a few functions to create a dataset with the right dimensions for the rnn model
# It uses the impl module to obtain the array of feature vectors
# It also creates an array for unicode Tamil character sequences that will act as y-labels (for decoder)

#%% imports

import impl
import os
import numpy as np
path = ("C:/Users/Sumangala/Desktop/OHWR/Dataset_TamilOHR" 
        + "/Testing/Tamil/WordLevel/hpl-tamil-iso-word-online/Tamil_Annotated_Words/set01/")
#%% conv2unicode

def getLabelsDict():
    m = 0
    wordencdict = {}
    train_y = []
    for folder in os.listdir(path):
        num  = folder.replace('usr','')
        f = open(path + folder + '/details_set1_usr{}.txt'.format(num))
        print('Collecting from folder {}'.format(folder))
        lines = f.readlines()
        l = []
        for line in lines:
            if line.startswith('.WORD LABEL'):
                charlist = []
                words = line.split(' ')
                tamilword = words[3]
                #print(tamilword)
                uni = tamilword.decode('utf-8')
                m = max(m,len(uni))
                for ch in uni:
                    #print(ch),
                    charlist.append(ch)
                l.append(charlist)
        #print(l)
        wordencdict[folder] = l
        train_y.append(l)
    return wordencdict, m
    
#%%

def getData():
    labelsdict, maxoutlen = getLabelsDict()
    strokedict = impl.collectData()
    train_x = []
    train_y = []
    for folder in os.listdir(path):
        i=0
        if i < 10:
            filename = '00000' + str(i) + 't0'
            filename += str(i%2 + 1)
            '''
            if i%2:
                filename += '2'
            else:
                filename += '1'
            '''
        for charlist in labelsdict[folder]:
            train_y.append(np.array(charlist))
            train_x.append(strokedict[folder+filename])
    data = np.stack([train_x,train_y],axis=0)
    return data
            
    












