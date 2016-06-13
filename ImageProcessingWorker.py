import zerorpc
import urllib2
import ntpath
import requests
from mimetypes import MimeTypes

import os
import signal, gevent
from files import ImageFile
from processor import (GrayScaleProcessor, BlurProcessor, SobelProcessor, 
                       ThresholdProcessor, MorphologyProcessor, HistogramEqualizationProcessor,
                      CannyProcessor, ScaleProcessor, HoughLinesProcessor,
                      SaveImageProcessor, InvertProcessor, RetrieveImageProcessor)
from segment_processor import (HierarchyContourSegmenter, LargeSegmentFilter, 
                               SmallSegmentFilter, LargeAreaSegmentFilter, ContainedSegmentFilter, UniqueSegmentFilter,
                              RatioSegmentFilter)
from feature_extractor import (SimpleFeatureExtractor, ThreeChannelsFeatureExtractor)
from segment_classifier import DeepCnnSegmentClassifier, CnnSegmentClassifier, DeepCnnSigmoidSegmentClassifier
from processor_stack import ProcessorStack
from power_number_reader import PowerNumberReader

import cv2
import itertools
import numpy as np
import sys

sys.path.append('/home/goldit/FRCN_ROOT/tools');
sys.path.append('/home/goldit/FRCN_ROOT/caffe-fast-rcnn/python');
sys.path.append('/home/goldit/FRCN_ROOT/lib');
sys.path.append('/home/goldit/FRCN_ROOT/lib/utils');
from ObjectDetector import ObjectDetector
from ObjectDetector import ObjectDetectorConfigs
adaptiveBlockSizes = [9, 17, 31]
erodes = [(0, 0), (3, 3), (3, 3), (5, 5), (7, 7)]
dilates = [(0, 0),(3, 3), (7, 3), (5, 3), (7, 3)]
class ImageProcessingWorker(object):
    def detect_digits(self, image):
        segments = np.empty( shape=(0, 0) )
        for adaptiveBlockSize, ((erode_x, erode_y), (dilate_x, dilate_y))  in itertools.product(adaptiveBlockSizes, zip(erodes, dilates)):
            digitDetectorStack = [ScaleProcessor(maxHeight=800, maxWidth=800),
                 SaveImageProcessor(),
                 GrayScaleProcessor(),
                 BlurProcessor(blur_x=5, blur_y=5),
                 HistogramEqualizationProcessor(HistogramEqualizationProcessor.CLAHE),
                 ThresholdProcessor(ThresholdProcessor.ADAPTIVE, adaptiveBlockSize=adaptiveBlockSize),
                 MorphologyProcessor(shape=cv2.MORPH_RECT, ksize=(erode_x, erode_y), op = MorphologyProcessor.ERODE),
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
        print 'Done Segmentation!'
    
        segmentFilterStack = [UniqueSegmentFilter(threshold = 0.8),
                              ThreeChannelsFeatureExtractor(feature_size=32, extension = 1, 
                                                            method=ThreeChannelsFeatureExtractor.PAD_BACKGROUND ,display_features = False),      
                              DeepCnnSigmoidSegmentClassifier(
                                            trained_network_file='model_deep_cnn_sigmoid_output_30itrs_svhn_manual_rotation.npz', 
                                            n_output = 11),
                              PowerNumberReader(bank_extension = 0, 
                                              bank_cut_off_ratio = 0.3, 
                                              height_error_acceptance = 0.7, 
                                              distance_error_acceptance = 0.3,
                                              residual_threshold = 2.5),
                             ]
        segmentFilter = ProcessorStack(segmentFilterStack)
        segmentFilter.process(image, segments = segments)
        return segmentFilter.get_output()['image']
    
    def upload_file (self, filename, url):
        files = {'file': (filename, open(filename, 'rb'), MimeTypes().guess_type(filename)[0])}
        response = requests.post(url, files=files)
        return response
    
    def download_file (self, filename, url):
        u = urllib2.urlopen(url)
        with open(filename, "wb") as local_file:
            local_file.write(u.read())

    def read_digit(self, link):
        try:
	    print "---------Digit Reader Task--------"
            print "receive request:" + link
            print "requesting link:" + 'http://'+serverIP+link
            filename = ntpath.basename(link)
            self.download_file(filename, url = 'http://'+serverIP+link)
            image = cv2.imread(filename)
            processed_image = self.detect_digits(image)
            cv2.imwrite(filename, processed_image)
            response = self.upload_file (filename, url = 'http://'+serverIP+'/api/user/upload/processed_digit_image')
	    print "Done"
            #os.remove(filename);
            if response.status_code!= 200:
                return -2
            else:
                return 0
        except Exception, err:
            print Exception, err
            return -1

    def recognize_object(self, link):
        try:
	    print "---------Object Recognizer Task--------"
            print "receive request:" + link
            print "requesting link:" + 'http://'+serverIP+link
            filename = ntpath.basename(link)
            self.download_file(filename, url = 'http://'+serverIP+link)
	    result = objDetector.detect(filename)
	    #result = '<map id=\"imgmap20163291213\" name=\"imgmap20163291213\"><area shape=\"rect\" alt=\"brown-disc-4\" title=\"\" coords=\"326,398,363,447\" href=\"\" target=\"\" /><area shape=\"rect\" alt=\"brown-disc-4\" title=\"\" coords=\"547,427,580,470\" href=\"\" target=\"\" /><area shape=\"rect\" alt=\"top-1\" title=\"\" coords=\"571,441,594,466\" href=\"\" target=\"\" /><area shape=\"rect\" alt=\"brown-disc-4\" title=\"\" coords=\"777,444,811,487\" href=\"\" target=\"\" /><area shape=\"rect\" alt=\"pole-top-T-1\" title=\"\" coords=\"314,392,829,634\" href=\"\" target=\"\" /><!-- Created for Drone Project--></map>'
	    filename = filename + ".map"
	    with open(filename, "w") as map_file:
	      map_file.write(result)
            #image = cv2.imread(filename)
            #processed_image = self.detect_digits(image)
            #cv2.imwrite(filename, processed_image)
            response = self.upload_file (filename, url = 'http://'+serverIP+'/api/user/upload/processed_drone_object_map')
	    print "Done"
            #os.remove(filename);
            if response.status_code!= 200:
                return -2
            else:
                return 0
        except Exception, err:
            print Exception, err
            return -1

        
def on_exit(signal, frame=None):
    print('You pressed Ctrl+C!')
    r = requests.post('http://'+serverIP+'/api/worker/remove', params = {'desc':'Server with TITAN X GPU'})
    if (r.status_code == 200):
        print "OK: Removing from server"
    else:
        print "Fail: Removing from server"
    sys.exit(0);


serverIP = '192.168.1.126';
if __name__ == "__main__":
    r = requests.post('http://'+serverIP+'/api/worker/register', params = {'desc':'Server with TITAN X GPU'})
    if (r.status_code== 200):
        print "OK: Registering"
        s = zerorpc.Server(ImageProcessingWorker(), heartbeat = 20)
        s.connect("tcp://"+serverIP+":3000")
        gevent.signal(signal.SIGINT, on_exit)
        #win32api.SetConsoleCtrlHandler(on_exit, 1)
        print "Start server"
	objDetector = ObjectDetector('VGG16')
        s.run()
    else:
        print "Fail: Registering"
