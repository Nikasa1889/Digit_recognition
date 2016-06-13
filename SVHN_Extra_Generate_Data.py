import scipy.io as sio
import numpy as np
import itertools
import cv2
import pickle
from feature_extractor import ThreeChannelsFeatureExtractor
from processor_stack import ProcessorStack
from processor import (GrayScaleProcessor, BlurProcessor, SobelProcessor, 
                       ThresholdProcessor, MorphologyProcessor, HistogramEqualizationProcessor,
                      CannyProcessor, ScaleProcessor, HoughLinesProcessor,
                      SaveImageProcessor, InvertProcessor, RetrieveImageProcessor)
from segment_processor import (HierarchyContourSegmenter, LargeSegmentFilter, 
                               SmallSegmentFilter, LargeAreaSegmentFilter, ContainedSegmentFilter, UniqueSegmentFilter,
                              RatioSegmentFilter)

from feature_extractor import (SimpleFeatureExtractor, ThreeChannelsFeatureExtractor)

#Box is represented by x, y, w, h. x, y is top left.
def readDigitStruct (digitStruct):
    #Return name, and an array of boxes
    name = digitStruct[0][0]
    boxes = digitStruct[1][0]
    segments = []
    segment_types = []
    for i in range(len(boxes)):
        box = boxes[i]
        x = box[1][0][0]
        y = box[2][0][0]
        h = box[0][0][0]
        w = box[3][0][0]
        segment = (x, y, w, h)
        segment_type = box[4][0][0]
        segments.append(segment)
        segment_types.append(segment_type)
    segment_types = np.array(segment_types).astype(int)
    segment_types[segment_types == 10] = 0
    return (name, np.array(segments).astype(int), segment_types)

def processImage (image, segments, segment_types):
    featureExtractStack = [ThreeChannelsFeatureExtractor(feature_size=32, extension = 1, 
                                                    method=ThreeChannelsFeatureExtractor.PAD_BACKGROUND ,display_features = True)]
    featureExtractor = ProcessorStack(featureExtractStack)
    featureExtractor.process(image, segments = segments, 
                               segment_types = segment_types)
    #featureExtractor.display()
    regions, labels = featureExtractor.get_output()['regions'], featureExtractor.get_output()['segment_types']
    return (regions, labels)
    

import sys
imDir = sys.argv[1]
print 'Generating data from: ', imDir

boxes = sio.loadmat(imDir + 'digitStruct_v7.mat')
boxes = boxes['digitStruct']
n = boxes.shape[1]

print "Total Images:", n
X_train = []
y_train = []
for i in range(n):
    imName, segments, segment_types = readDigitStruct(boxes[0, i])
    image = cv2.imread(imDir + imName)
    if image is not None:
        regions, labels = processImage (image, segments, segment_types)
        X_train.append(regions)
        y_train.append(labels)
        
    if ((i+1)%10000 == 0) or (i+1 == n):
        print i
        sys.stdout.flush()
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_train = np.rollaxis(X_train, 3, 1)
        np.savez(imDir + "total_extra" + str(i)+'.npz', X_train=X_train, y_train = y_train)
        X_train = []
        y_train = []
    
    
print "Total digits: ", len(X_train)

