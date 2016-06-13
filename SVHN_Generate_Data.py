import scipy.io as sio
import numpy as np
import itertools
import cv2
import pickle
import sys
from feature_extractor import ThreeChannelsFeatureExtractor
from processor_stack import ProcessorStack
from processor import (GrayScaleProcessor, BlurProcessor, SobelProcessor, 
                       ThresholdProcessor, MorphologyProcessor, HistogramEqualizationProcessor,
                      CannyProcessor, ScaleProcessor, HoughLinesProcessor,
                      SaveImageProcessor, InvertProcessor, RetrieveImageProcessor)
from segment_processor import (HierarchyContourSegmenter, LargeSegmentFilter, 
                               SmallSegmentFilter, LargeAreaSegmentFilter, ContainedSegmentFilter, UniqueSegmentFilter,
                              RatioSegmentFilter)

from segment_classifier import NonDigitSegmentClassifier
from feature_extractor import (SimpleFeatureExtractor, ThreeChannelsFeatureExtractor)
adaptiveBlockSizes = [9, 17, 31]
erodes = [(0, 0), (3, 3), (3, 3), (5, 5), (7, 7)]
dilates = [(0, 0),(3, 3), (7, 3), (5, 3), (7, 3)]

imDir = 'J:/SVHN/test/'

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

def propose_segments(image):
    segments = np.empty( shape=(0, 0) )
    for adaptiveBlockSize, ((erode_x, erode_y), (dilate_x, dilate_y))  in itertools.product(adaptiveBlockSizes, zip(erodes, dilates)):
        digitDetectorStack = [#ScaleProcessor(maxHeight=800, maxWidth=800),
             SaveImageProcessor(),
             GrayScaleProcessor(),
             BlurProcessor(blur_x=5, blur_y=5),
             HistogramEqualizationProcessor(HistogramEqualizationProcessor.CLAHE),
             ThresholdProcessor(ThresholdProcessor.ADAPTIVE, adaptiveBlockSize=adaptiveBlockSize), # 31 for normal. Change this to catch digits!
             MorphologyProcessor(shape=cv2.MORPH_RECT, ksize=(erode_x, erode_y), op = MorphologyProcessor.ERODE), #Change this
             MorphologyProcessor(shape=cv2.MORPH_RECT, ksize=(dilate_x, dilate_y), op = MorphologyProcessor.DILATE),
             HierarchyContourSegmenter(),
             RetrieveImageProcessor(),
             LargeSegmentFilter(min_width=4, min_height=13),
             SmallSegmentFilter(max_width=100, max_height=200),
             LargeAreaSegmentFilter(min_area = 120),
             RatioSegmentFilter(min_h_w_ratio = 1, max_h_w_ratio = 4.5)
             ] 
        digitDetector = ProcessorStack(digitDetectorStack)
        digitDetector.process(image)
        #digitDetector.display()
        output = digitDetector.get_output()
        if len(segments) == 0:
            segments = output['segments']
            image = output['originalImg']
        else:
            segments = np.concatenate((segments, output['segments']), axis = 0)
    
    segmentFilterStack = [UniqueSegmentFilter(threshold = 0.7),
                          ThreeChannelsFeatureExtractor(feature_size=32, extension = 1, 
                                                        method=ThreeChannelsFeatureExtractor.PAD_BACKGROUND ,display_features = False)
                         ]
    segmentFilter = ProcessorStack(segmentFilterStack)
    segmentFilter.process(image, segments = segments)
    return segmentFilter

def processImage (image, segments, segment_types):
    #featureExtractStack = [ThreeChannelsFeatureExtractor(feature_size=32, extension = 1, 
    #                                                method=ThreeChannelsFeatureExtractor.PAD_BACKGROUND ,display_features = True)]
    #featureExtractor = ProcessorStack(featureExtractStack)
    
    #featureExtractor.process(image, segments = segments, segment_types = segment_types)
    #cv2.namedWindow( "Original", cv2.WINDOW_AUTOSIZE );
    #cv2.imshow( "Original", image );    
    #featureExtractor.display()
    segmentProposer = propose_segments(image)
    non_digit_segments = segmentProposer.get_output()['segments']
    
    nonDigitClassifyStack = [NonDigitSegmentClassifier(loose_high_threshold = 0.75, loose_low_threshold = 0.15, occupy_threshold = 0.8),
                            ThreeChannelsFeatureExtractor(feature_size=32, extension = 1, 
                                                    method=ThreeChannelsFeatureExtractor.PAD_BACKGROUND ,display_features = True)]
    nonDigitClassifier = ProcessorStack(nonDigitClassifyStack)
    nonDigitClassifier.process(image, digit_segments = segments, 
                               digit_segment_types = segment_types, 
                               non_digit_segments = non_digit_segments)
    #nonDigitClassifier.display()
    regions, labels = nonDigitClassifier.get_output()['regions'], nonDigitClassifier.get_output()['segment_types']
    return (regions, labels)
    

#with open(imDir +'boxes.pkl', 'rb') as input:
#    boxes = pickle.load(input)
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
        
    if ((i+1)%1000 == 0) or (i+1 == n):
        print i
        sys.stdout.flush()
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_train = np.rollaxis(X_train, 3, 1)
        print X_train.shape
        
        np.savez(imDir + "total_" + str(i)+'.npz', X_train=X_train, y_train = y_train)
        X_train = []
        y_train = []
