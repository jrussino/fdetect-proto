fdetect-proto
=============

Author: Joseph A. Russino (jrussino@gmail.com)

This is a prototype implementation of face detection via cascade classifier.
It was written primarily as an exercise to demonstrate understanding of the Viola-Jones algorithm,
and as such is not intended for real-world use (it runs much too slowly).
Requires OpenCV 2.4.9, which is used for convenient image loading, etc. (but NOT for the actual detection).
It also leverages OpenCV's pre-trained classifier .xml files, so that I could focus on implementation rather than training.

Included files:
cascadeClassifier.py - the cascade clasifier implementation
detectFaces.py - script for running detection on an image
lbpcascade_frontalface.xml - OpenCV Local Binary Patterns cascade file

