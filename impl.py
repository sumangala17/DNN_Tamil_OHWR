# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:26:46 2019

@author: Sumangala
"""

# pre-processing function implementations

# This module primarily generates the feature vectors for strokes, in addition to a few correcting functions
# Imports module datastr to access stroke characteristics
# This is next in the heirarchy after datastr, and the last module to use stroke point data directly

#%% imports

import datastr as ds
import numpy as np
import os

HOTHRESHOLD = 0.2


def horizontalOverlap(inputStroke1, inputStroke2):
    if inputStroke1.xMin < inputStroke2.xMin:
        ho1 = float(inputStroke1.xMax - inputStroke2.xMin) / (inputStroke1.xMax - inputStroke1.xMin)
        ho2 = float(inputStroke1.xMax - inputStroke2.xMin) / (inputStroke2.xMax - inputStroke2.xMin)
        return ho1 if ho1 > ho2 else ho2
    else:
        ho1 = float(inputStroke2.xMax - inputStroke1.xMin) / (inputStroke1.xMax - inputStroke1.xMin)
        ho2 = float(inputStroke2.xMax - inputStroke1.xMin) / (inputStroke2.xMax - inputStroke2.xMin)
        return ho1 if ho1 > ho2 else ho2

'''
def findNearestStroke(inputSG, pos):
    nearest = 1
    distMin = sys.float_info.max
    if inputSG == None or pos > inputSG.numOfStrokes:
        return 1
    inputStroke = inputSG.strokeGroup[pos]
    for i in range(pos-1, 0, -1):
        currentStroke = ds.OHRStroke()
        currentStroke = inputSG.strokeGroup[i]
        
        dx2 = pow(currentStroke.xMean - inputStroke.xMean, 2)
        dy2 = pow(currentStroke.yMean - inputStroke.yMean, 2)
        dist  = sqrt(dx2 + dy2)
        if dist < 0.85 * distMin:
            distMin = dist
            nearest = i
    return nearest
'''

        
'''
        for i in range(pos-1, len(self.strokeGroup)):
            temp.append(self.strokeGroup[i])
        del self.strokeGroup[pos-1:]
        self.strokeGroup.append(stroke)
        
        for i in range(len(temp)):
            self.strokeGroup.append(temp[i])
'''


def delayedStroke(inputSG):     #input is of type 
    outputSG = ds.OHRStrokeGroup()
    para = np.zeros((30,2))
    #idx = np.zeros((30,1))
    idx = [i for i in range(1, len(inputSG.strokeGroup))]
    for i in range(1,len(inputSG.strokeGroup)):
        para[i][0] = inputSG.strokeGroup[i].xMin
        para[i][1] = inputSG.strokeGroup[i].xMax
        #idx[i] = i
    
    for i in range(1,len(inputSG.strokeGroup)):
        for j in range(1,len(inputSG.strokeGroup - 1)):
            if para[j][0] > (para[j+1][1] + (0.1 * (para[j+1][1] - para[j+1][0]))) and horizontalOverlap(inputSG.strokeGroup[idx[j]], inputSG.strokeGroup[idx[j]]) < HOTHRESHOLD:
                
                #nearest = findNearestStroke(inputSG,idx[j])
                #nearestStroke = inputSG.strokeGroup[nearest]
                
                para[j][0], para[j+1][0] = para[j+1][0], para[j][0]
                para[j][1], para[j+1][1] = para[j+1][1], para[j][1]
                idx[j], idx[j+1] = idx[j+1], idx[j]
                
            
    for i in range(len(inputSG.strokeGroup)):
        currentStroke = inputSG.strokeGroup[idx[i]]
        outputSG.addStroke(currentStroke, i)

    return outputSG
            
        

def collectData():
    base_path = "C:/Users/Sumangala/Desktop/OHWR/Dataset_TamilOHR"
    train_path_word = base_path + "/Testing/Tamil/WordLevel/hpl-tamil-iso-word-online/Tamil_Annotated_Words/set01/"
    #train_path = "C:/Users/Sumangala/Desktop/OHWR/Dataset_TamilOHR/Training/Tamil/HPL_org"
    contents = os.listdir(train_path_word)
    
    strokedict = {}
    
    i = 0
    count = 0
    #map stroke groups to characters for training
    for folder in contents:
        if not folder.startswith("usr"):    #iterate over the folders only
            continue
        i += 1
        if i > 1:
            break
        print('Now collecting from {}'.format(folder))
        count = 0
        
        for filename in os.listdir(train_path_word + "/" + folder):
            count += 1
            if count > 1:
                break
            strokedict[folder + filename] = []
            fname = train_path_word + "/" + folder + "/" + filename
            stroke_group = ds.OHRStrokeGroup()
            stroke_group.readFromFile(fname)
            stroke_group = stroke_group.resample(15*len(stroke_group.strokeGroup))
            stroke_group.plotStrokeGroup()
            stroke_group.normalize()
            stroke_group.plotStrokeGroup()
            
            fv = np.zeros((stroke_group.numOfStrokes, 20),dtype = complex)
            for i in range(len(stroke_group.strokeGroup)):
                fvec = stroke_group.strokeGroup[i].featureVector()
                fv[i,:] = fvec
                
            print(fv)
            strokedict[folder + filename].append(fv)
            
            #train_x = np.stack((train_x, fv), axis = 0)
    
            #train_x = np.array(fv)
    return strokedict
   # print(train_x)
    #print(train_x.shape)
            
    #clf = SVC(gamma = 'scale')
    #clf.fit(train_x, train_y)
    




