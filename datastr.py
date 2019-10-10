# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 20:45:42 2019

@author: Sumangala
"""

# This is the most basic module defining all classes used for stroke data collection
# No import of other modules required for this
# Run these cells first while testing program cell wise

#%% define classes
import matplotlib.pyplot as plt
import copy
import numpy as np
import math


class OHRCoord:
    def __init__(self,_x = 0,_y = 0):   #sets to 0 if no parameters supplied
        self.x = _x
        self.y = _y
        
        
#_______________________________________________  OHR Stroke  _________________________________________________
    

class OHRStroke:
    
    def __init__(self):
        self.stroke = []    #list of OHRCoord
        self.xMax = 0
        self.xMin = 100000
        self.yMax = 0
        self.yMin = 100000
        self.xMean = 0
        self.yMean = 0
        self.vLinear = 0
        self.arcLength = 0
        self.numberOfPoints = 0
        self.dot = False
        #self.padam = False
        #self.horizonLine = False
        
    def featureVector(self):
    
        aspect_ratio = (self.xMax - self.xMin) / (self.yMax - self.yMin)
        feature_vector = [self.arcLength, self.xMean, self.yMean, aspect_ratio]
        labels = []
        for i in range(len(self.stroke)):
            coord = self.stroke[i]
            labels.append(complex(coord.x, coord.y))
            
        labels = np.fft.fft(labels)
        if len(labels) > 16:
            labels = labels[:16]
        else:
            l = len(labels)
            labels = np.append(labels,[0]*(16-l))

        feature_vector.extend(labels)
        feature_vector = np.array(feature_vector)
        return feature_vector

        
    def display(self):
        print('Stroke coordinates:')
        for crd in self.stroke:
            print(crd.x, crd.y)
        print('end of display stroke')
            
    
    def update(self):
        self.numberOfPoints = len(self.stroke)
        if self.numberOfPoints == 0:
            return
        self.xMax = 0
        self.xMin = 100000
        self.yMax = 0
        self.yMin = 100000
        
        for coord in self.stroke:
            self.xMax = max(self.xMax, coord.x)
            self.xMin = min(self.xMin, coord.x)
            self.yMax = max(self.yMax, coord.y)
            self.yMin = min(self.yMin, coord.y)
           
        self.arcLength = 0
        for i in range(len(self.stroke)-1):
            self.arcLength += distanceCartesian(self.stroke[i], self.stroke[i+1])
        
        xsum = 0
        ysum = 0
        
        for crd in self.stroke:
            #crd = OHRCoord(crd.x, crd.y)
            xsum += crd.x
            ysum += crd.y
        self.xMean = float(xsum)/self.numberOfPoints
        self.yMean = float(ysum)/self.numberOfPoints
        
        #to avoid divide by 0 error in normalizing and finding slope
        if self.xMax == self.xMin:
            self.xMax += 1
        if self.yMax == self.yMin:
            self.yMax += 1
        
        l = self.arcLength / distanceCartesian(self.stroke[0], self.stroke[len(self.stroke)-1])
        slope = float(self.yMax - self.yMin) / (self.xMax - self.xMin)
        
        if self.arcLength >= 5 and l <= 1.5 and slope >= 1.73:
            self.vLinear = 1 
        else:
            self.vLinear = 0
        
        '''
        print('end of update')
        print(self.xMax, self.xMin)
        print(self.yMax, self.yMin)'''
            
    def resample(self, numpoints):
        
        #print(self.numberOfPoints)
        out = OHRStroke()
        if numpoints == 0:
            numpoints = 1
            
        if numpoints == 1 or self.arcLength == 0:
            coord = self.stroke[0]
            out.stroke.append(coord)
            out.update()
            return out
        
        distance = self.arcLength / (numpoints - 1)
        distRem = self.arcLength / (numpoints - 1)
        
        pc = 0
        j = 0
        
        coord = self.stroke[0]
        
        while numpoints > 0:
            
            if pc == 0:
                resampled = coord
                out.stroke.append(resampled)
                present = OHRCoord(resampled.x, resampled.y)
                pc = pc + 1
                numpoints -= 1
                j += 1
                coord = self.stroke[j]
                
            elif numpoints == 1 and coord == self.stroke[self.numberOfPoints - 1]:
                resampled = coord
                out.stroke.append(resampled)
                numpoints -= 1
                
            else:
                distCov = distanceCartesian(coord, present)
                if distCov < distRem:
                    distRem -= distCov
                    present = coord
                    j = j + 1 
                    coord = self.stroke[j]

                else:
                    resampled.x = ((1 - distRem/distCov)*present.x) + (distRem/distCov)*coord.x
                    resampled.y = ((1 - distRem/distCov)*present.y) + (distRem/distCov)*coord.y
                    out.stroke.append(copy.deepcopy(resampled))
                    present = OHRCoord(resampled.x, resampled.y)
                    distRem = distance 
                    pc += 1
                    numpoints -= 1
         
        out.update()
        return out
        
    
    def addCoord(self, coord):
        self.stroke.append(coord)
        self.update()
        
    def getCoord(self, N):
        if N <= self.numberOfPoints:
            return self.stroke[N-1]
        else:
            print('Invalid index')
            return None
    '''
    def removeCoord(self, N):
        if N <= self.numberOfPoints:
            #delete from start to Nth element in stroke 
            del self.stroke[:N-1]
        else:
            print('Invalid stroke index')
        self.update()
        return           
    '''         
#__________________________________________________  OHR Stroke Group  ________________________________________


def distanceCartesian(coord1, coord2):
    dx2 =  (coord1.x - coord2.x)**2  #pow(coord1.x - coord2.x, 2)
    dy2 = (coord1.y - coord2.y)**2   #pow(coord1.y - coord2.y, 2)
    ans  = math.sqrt(dx2 + dy2)
    #print(ans, coord2.x, coord2.y)
    return ans if ans != 0 else 1
 
           
class OHRStrokeGroup:
    def __init__(self):
        self.strokeGroup = []   # list of OHR Strokes
        self.padamStroke = None
        self.dotStroke = None
        self.numOfDots = 0
        self.numOfStrokes = 0
        self.numOfDominantPoints = 0
        self.hasPadam = False
        self.isConjunct = False
        self.vMin = None
        self.length = 0.0
        self.xMin = 100000
        self.yMin = 100000
        self.xMax = self.yMax = self.xMean = self.yMean = self.v = 0.0
        self.aspectRatio = 0.0
        self.SVM_label = None
        self.sg_name = None
        
        
    def sgFeatureVector(self):
        featureVector = np.array([self.aspectRatio, self.length, self.numOfStrokes])
        for stk in self.strokeGroup:
            fv = stk.featureVector()
            np.concatenate((featureVector, fv))
        return featureVector
    
    def addStroke(self, stroke, pos):
        if pos < self.numOfStrokes:        
            self.strokeGroup.insert(pos,stroke)
        elif pos == self.numOfStrokes + 1:
            self.strokeGroup.append(stroke)
        else:
            print('invalid stroke index')
        
        self.update()            
            
        
    def plotStrokeGroup(self):
        colour = ['b','r', 'g', 'y','c','m','k','w']
        i = 0
        for stroke in self.strokeGroup:            
            xval = []
            yval = []
            for crd in stroke.stroke:
                xval.append(crd.x)
                yval.append(crd.y)
            ax=plt.gca()                            # get the axes
            ax.invert_yaxis()       #invert y axis
            #ax.invert_xaxis()
            plt.plot(xval,yval, colour[i%8] + 'o')
            i += 1
        plt.show()
        
    def readFromFile(self, filename):
        
        f = open(filename, 'r')
        lines = f.readlines()
        coord = OHRCoord()
        penUp = penDown = 0
        
        for line in lines:
            if line.startswith('.PEN_DOWN'):
                penDown = 1
                penUp = 0
                stroke = OHRStroke()
                self.numOfStrokes += 1
            elif line.startswith('.PEN_UP'):
                penDown = 0
                penUp = 1
                stroke.update()
                self.strokeGroup.append(stroke)
            elif penDown == 1 and penUp == 0:
                words = line.split(' ')
                coord = OHRCoord(int(words[0]), int(words[1]))
                #coord.x = int(words[0])
                #coord.y = int(words[1])
                stroke.addCoord(coord)

        f.close()
        
        self.update()
        return
    
    
    def display(self):
        for stroke in self.strokeGroup:
            stroke.display()
            
    
    def update(self):
        numPoints = 0
        self.numOfStrokes = len(self.strokeGroup)
        if self.numOfStrokes == 0:
            return
        
        self.xMean = 0.0
        self.yMean = 0.0
        self.xMax = 0
        self.xMin = 10000
        self.yMax = 0
        self.yMin = 10000
        
        for stroke in self.strokeGroup:
            self.xMax = max(stroke.xMax, self.xMax)
            self.xMin = min(stroke.xMin, self.xMin)
            self.yMax = max(stroke.yMax, self.yMax)
            self.yMin = min(stroke.yMin, self.yMin)
            numPoints += stroke.numberOfPoints
            self.xMean += stroke.xMean * stroke.numberOfPoints
            self.yMean += stroke.yMean * stroke.numberOfPoints
            self.length += stroke.arcLength
        
        self.xMean = float(self.xMean)/numPoints
        self.yMean = float(self.yMean)/numPoints
    
        if self.yMax == self.yMin:
            self.yMax += 1
        if self.xMax == self.xMin:
            self.xMax += 1
            
        self.aspectRatio = float(self.yMax - self.yMin) / (self.xMax - self.xMin)
        
    
    def resample(self, numPoints = 64):
        
        strokeCount = self.numOfStrokes + 1
        strokeCount -= 1
        pointsCovered = 0
        out = OHRStrokeGroup()
        for stroke in self.strokeGroup:
            
            stk = OHRStroke()
            strokeCount -= 1
            if strokeCount > 0:
                numPointsStroke = int(numPoints * float(stroke.arcLength) / self.length)
                if ((numPoints * float(stroke.arcLength)/self.length) - numPointsStroke) > 0.5:
                    numPointsStroke += 1
                if numPointsStroke == 0:
                    numPointsStroke = 1
                
                if (numPoints - pointsCovered - numPointsStroke) < strokeCount:
                    numPointsStroke = numPoints - pointsCovered - strokeCount
                pointsCovered += numPointsStroke
            else:
                numPointsStroke = numPoints - pointsCovered
            stk = copy.deepcopy(stroke.resample(numPointsStroke))
            if stk == None:
                continue
            
            out.strokeGroup.append(copy.deepcopy(stk))
        
        out.update()
        return out
        
    
    def normalize(self):
        
        if self.xMax == self.xMin:
            self.xMax += 1
        if self.yMax == self.yMin:
            self.yMax += 1
            
        xDiff = self.xMax - self.xMin
        yDiff = self.yMax - self.yMin
        
        #print(self.xMax, self.xMin, self.yMax, self.yMin)
        
        for ohrstroke in self.strokeGroup:
            for coord in ohrstroke.stroke:
                coord.x = round(float(coord.x - self.xMin) / xDiff,4)
                coord.y = round(float(coord.y - self.yMin) / yDiff,4)
        
        self.update()
        
'''        
train_path = "C:/Users/Sumangala/Desktop/OHWR/Dataset_TamilOHR/Training/Tamil/HPL_org/usr_17/005t06.txt"
stroke_group = OHRStrokeGroup()
stroke_group.readFromFile(train_path)
stroke_group.plotStrokeGroup()
stroke_group = stroke_group.resample()
stroke_group.plotStrokeGroup()       
stroke_group.normalize()
stroke_group.plotStrokeGroup()
'''     
            

            
            
            