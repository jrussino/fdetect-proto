#!/usr/bin/env python
#
# file:         detectFaces.py
# author:       Joseph A. Russino (jrussino@gmail.com)
# date:         2014/28/08
#
# description:  run face detection on a single image

import argparse
import cascadeClassifier as cc
import cv2
import os
import sys


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


if __name__ == '__main__' :
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Detect faces in an image')
    parser.add_argument('imageFile', type=str, help='Input image on which to run face detection.') 
    parser.add_argument('-w', '--withOpenCV', action='store_true', help='Run opencv face detection (for comparison).')
    parser.add_argument('-o', '--outDir', type=str, default='./', help='Output directory.')
    parser.add_argument('-c', '--classifierFile', type=str, default='./lbpcascade_frontalface.xml', help='Cascade classifier xml file.')
    args = parser.parse_args()

    if not os.path.exists(args.imageFile) :
        print 'Could not open image file: ' + args.imageFile
        sys.exit(1)

    # Load classifier
    print "loading classifier..."
    myCascade = cc.CascadeClassifier(args.classifierFile)
    print "\tDONE!"
 
    # Load image, convert to grayscale, and equalie
    rawImage  = cv2.imread(args.imageFile)
    grayImage = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.equalizeHist(grayImage)

    # Generate integral image and run cascade classifier
    print "runnning detection..."
    integralImage = cc.integralImage(grayImage)
    detections = myCascade.detectMultiScale(integralImage, 1.3, 2)
    print "\tDONE!"

    # Output results and save
    print str(len(detections)) + " total detections"  
    vis = rawImage.copy()
    if len(detections) > 0 :
        detections = [[x,y,x+w,y+h] for [x,y,w,h] in detections]
    draw_rects(vis, detections, (0, 255, 0))
    outFileBase = os.path.splitext(os.path.basename(args.imageFile))[0]
    outFile = os.path.join(args.outDir,outFileBase+'_detections.png')
    print 'writing output to ' + outFile
    cv2.imwrite(outFile, vis)

    if args.withOpenCV :
        # Load opencv classifier
        print "loading opencv classifier..."
        cvCascade = cv2.CascadeClassifier(args.classifierFile)
        print "\tDONE!"
        
        # Run opencv cascade classifier
        print "runnning opencv detection..."
        detectionsCV = cvCascade.detectMultiScale(grayImage, scaleFactor=1.3, minNeighbors=0, minSize=(24,24), flags = cv2.CASCADE_SCALE_IMAGE)
        print "\tDONE!"
        if len(detectionsCV) > 0 :
            detectionsCV[:,2:] += detectionsCV[:,:2]
        print str(len(detectionsCV)) + " total detections (opencv)"  
        vis = rawImage.copy()
        draw_rects(vis, detectionsCV, (0, 255, 0))
        outFileCV = os.path.join(args.outDir,outFileBase+'_detections_openCV.png')
        print 'writing output to ' + outFileCV
        cv2.imwrite(outFileCV, vis)

