#!/usr/bin/python
#
# file:         cascadeClassifier.py
# author:       Joseph A. Russino (jrussino@gmail.com)
# date:         2014/05/13
#
# description:  prototype implementation of cascade classifier

import cv2
import numpy as np
import sys
import xml.etree.cElementTree as et


# Cascade Classifier
class CascadeClassifier:
    # Initialization - load cascade from XML file
    def __init__(self, xmlFile) :
        # load cascade from xml
        tree = et.ElementTree(file=xmlFile)
        cascade = tree.find('cascade')

        # parse out general classifier info
        height = int(cascade.find('height').text)
        width = int(cascade.find('width').text)
        self.windowSize = (height,width)

        # parse out list of rectangles
        rectangles = []
        for element in cascade.iterfind('features/_/rect') :
            rect = strToIntList(element.text)
            rectangles.append(rect)

        # create each of the stages
        self.stages = []
        for element in cascade.iterfind('stages/_') :
            stage = CascadeStage(element, rectangles)
            self.stages.append(stage)
        return
  
    # Detect Multi Scale - run detection on an image at multiple scales
    def detectMultiScale(self, image, scaleFactor, stepSize=1) :
        detections = []
        maxScale = min([image.shape[0]/self.windowSize[0],      #NOTE: both inputs are int, so there's an implicit floor() here
                        image.shape[1]/self.windowSize[1]]) 
        scale = 1.0
        print 'max scale: ' + str(maxScale)
        while scale <= maxScale :
            print 'evaluating at scale: ' + str(scale) \
                  + '(' + str(int(round(scale*self.windowSize[0]))) \
                  + 'x' + str(int(round(scale*self.windowSize[1])))+ ')'
            for detection in self.detectSingleScale(image,scale,stepSize) :
                detections.append(detection)
            scale *= scaleFactor
        return detections

    # Detect Single Scale - run detection on an image at a single scale
    def detectSingleScale(self, image, scale, stepSize=1) :
        detections = []
        # subtracting 2 from each dimension of the scan to avoid overflowing past the edge of the image due to rounding
        for windowRow in xrange(0, image.shape[0]-int(round(self.windowSize[0]*scale))-2, stepSize) : 
            for windowCol in xrange(0, image.shape[1]-int(round(self.windowSize[1]*scale))-2, stepSize) :
                if self.detectAtLocation(image, (windowRow,windowCol), scale) :
                    # detection found; return [x y width height]
                    detectionLocation = [windowCol, 
                                    windowRow, 
                                    int(round(self.windowSize[0]*scale)), 
                                    int(round(self.windowSize[1]*scale))]
                    detections.append(detectionLocation)
        return detections
  
    # Detect At Location - run detection on an image at a single location
    def detectAtLocation(self, image, location, scale) :
        for stage in self.stages :
            if stage.evaluate(image, location, scale) == False :
                return False
        return True


# Cascade Classifier Stage
class CascadeStage :
    # Initialization - load stage from XML node
    def __init__(self, xmlNode, rectangles) :
        self.threshold = float(xmlNode.find('stageThreshold').text)

        # get all features
        self.features = []
        for element in xmlNode.iterfind('weakClassifiers/_') :
            feature = LBPFeature(element, rectangles)
            self.features.append(feature)
        return 

    # Evaluate - evaluate stage on an image at a specified location
    def evaluate(self, image, location, scale) :
        score = 0.0
        for feature in self.features :
            score += feature.evaluate(image, location, scale)
            if score < self.threshold :
                return False
        return True


# Local Binary Pattern Feature
class LBPFeature :
    # Initialization - load feature from XML node
    def __init__(self, xmlNode, rectangles) :
        internalNodes = strToIntList(xmlNode.find('internalNodes').text)
        leafValues = strToFloatList(xmlNode.find('leafValues').text)
        rectIndex = internalNodes[2]
        self.rectangle = rectangles[rectIndex]
        self.failWeight = leafValues[0]
        self.passWeight = leafValues[1]

        # construct lookup table
        self.lookupTable = []
        for node in internalNodes[3:] :
            self.lookupTable.append(node)
        return
  
    # Evaluate - evaluate feature on an image at a specified location
    def evaluate(self, image, location, scale) :
        windowRow = location[0]
        windowCol = location[1]
        rectCol = self.rectangle[0]
        rectRow = self.rectangle[1]
        rectW = self.rectangle[2]
        rectH = self.rectangle[3]

        rowStart = windowRow + int(round(rectRow*scale))
        colStart = windowCol + int(round(rectCol*scale))
        rectW = int(round(rectW*scale))
        rectH = int(round(rectH*scale))

        # compute the values for each box
        boxVals = []
        for i in xrange(0,3) :
            for j in xrange(0,3) :
                rowMin = rowStart + i*rectH
                rowMax = rowStart + (i+1)*rectH
                colMin = colStart + j*rectW
                colMax = colStart + (j+1)*rectW
                boxUL = image[rowMin][colMin]
                boxUR = image[rowMin][colMax]
                boxLL = image[rowMax][colMin]
                boxLR = image[rowMax][colMax]
                boxVal = boxLR + boxUL - boxUR - boxLL
                boxVals.append(boxVal)

        # compare to center box to get local binary pattern
        lbp = ''
        for boxIndex in [0,1,2,5,8,7,6,3] :
            lbp += str(int(boxVals[boxIndex] >= boxVals[4]))
        lbp = int(lbp,2) # convert from binary string to integer

        # look up match value in the feature's lookup table
        matchValue = self.lookupTable[lbp>>5] & (1 << (lbp & 31))

        # set feature score based on match value
        featureScore = self.passWeight if matchValue == 0 else self.failWeight

        return featureScore


# String To Integer List - convert string of ints to list of ints
def strToIntList(string) :
    intList = [int(x) for x in string.split(' ') if (x and x != '\n')]
    return intList


# String To Float List - convert string of floats to list of floats
def strToFloatList(string) :
    floatList = [float(x) for x in string.split(' ') if (x and x != '\n')]
    return floatList

# Integral Image - compute the integral of a grayscale image
def integralImage(image) :
    integralImage = np.zeros([x+1 for x in image.shape],dtype=np.int32)
    # rows
    for i in xrange(1,integralImage.shape[0]) :
        # columns
        for j in xrange(1,integralImage.shape[1]) :
            rowSum = integralImage[i-1,j]
            colSum = integralImage[i,j-1]
            diagSum = integralImage[i-1,j-1]
            integralImage[i][j] = image[i-1][j-1] + rowSum + colSum - diagSum 
    return integralImage 

