from processor import Processor, BrightnessProcessor
from utils import contained_segments_matrix, merge_dicts, draw_segments, show_image_and_wait_for_key
from tesseract_utils import segments_to_numpy

import cv2
import numpy
#Segment is a numpy array, with each row is (x,y,w,h)
#Segment_type is a numpy array
class SegmentProcessor( Processor ):
    '''Processor for segments, given image and segments, returning only the desirable
    ones'''
    def display( self, display_before=False):
        '''shows the effect of this filter'''
        try:
            image = self._input['originalImg'].copy()
        except KeyError:
            try:
                image = self._input['image'].copy()
            except KeyError:
                raise Exception("You need to set the _input['image'] or _input['originalImg'] attribute for displaying")
        #image= BrightnessProcessor(brightness=0.6).process( image )[0]
        if 'segments' in self._input:
            draw_segments( image, self._input['segments'], (255,0,0) )
        draw_segments( image, self._output['segments'], (0,255,0) )
        show_image_and_wait_for_key( image, "segments filtered by "+self.__class__.__name__)

class HierarchyContourSegmenter( SegmentProcessor ):
    def __init__( self ):
        pass
    def _process( self, image, **args ):
        image_input = image
        image = image.copy()
        contours,hierarchy = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        segments= segments_to_numpy( [cv2.boundingRect(c) for c in contours] )
        if len(segments)<=1:
            segments=numpy.empty((0, 4), dtype=int)
        self.contours, self.hierarchy= contours, hierarchy #store, may be needed for debugging
        return (image_input, merge_dicts(args, {'segments':segments}))

class LargeSegmentFilter( SegmentProcessor ):
    '''desirable segments are larger than some width or height'''
    def __init__(self, min_width=0.1, min_height=0.1):
        self.min_width = min_width
        self.min_height = min_height
    def _process( self, image, **args ):
        assert 'segments' in args
        segments = args['segments']
        if len(segments) == 0:
            return(image, args)
        good_width=  segments[:,2] >= self.min_width
        good_height= segments[:,3] >= self.min_height
        result = segments[good_width * good_height] #AND
        return (image, merge_dicts(args, {'segments':result}))  

class SmallSegmentFilter( SegmentProcessor ):
    '''desirable segments are smaller than some width or height'''
    
    def __init__(self, max_width=30, max_height=50):
        self.max_width = max_width
        self.max_height = max_height
    def _process( self, image, **args ):
        assert 'segments' in args
        segments = args['segments']
        if len(segments) == 0:
            return(image, args)
        good_width=  segments[:,2]  <= self.max_width
        good_height= segments[:,3] <= self.max_height
        result = segments[good_width * good_height] #AND
        return (image, merge_dicts(args, {'segments':result}))  
        
class LargeAreaSegmentFilter( SegmentProcessor ):
    '''desirable segments' area is larger than some'''
    def __init__(self, min_area=256):
        self.min_area = min_area
    def _process( self, image, **args ):
        assert 'segments' in args
        segments = args['segments']
        if len(segments) == 0:
            return(image, args)
        good_area = (segments[:,2]*segments[:,3]) >= self.min_area
        result = segments[good_area]
        return (image, merge_dicts(args, {'segments':result}))
    
class RatioSegmentFilter( SegmentProcessor ):
    '''desirable segments' area is larger than some'''
    def __init__(self,min_h_w_ratio = 1.2, max_h_w_ratio = 5):
        self.max_h_w_ratio = max_h_w_ratio
        self.min_h_w_ratio = min_h_w_ratio
    def _process( self, image, **args ):
        assert 'segments' in args
        segments = args['segments']
        if len(segments) == 0:
            return(image, args)
        good_min_ratio = segments[:, 3] >= self.min_h_w_ratio * segments[:, 2]
        good_max_ratio = segments[:, 3] <= self.max_h_w_ratio * segments[:, 2]
        result = segments[good_min_ratio * good_max_ratio]
        return (image, merge_dicts(args, {'segments':result}))

class ContainedSegmentFilter( SegmentProcessor ):
    '''desirable segments are not contained by any other'''
    def _process( self, image, **args ):
        assert 'segments' in args
        segments = args['segments']
        if len(segments) == 0:
            return(image, args)
        m= contained_segments_matrix( segments )
        no_contain = True - numpy.max(m, axis = 1)
        result = segments[no_contain]
        return (image, merge_dicts(args, {'segments':result}))

class UniqueSegmentFilter( SegmentProcessor ):
    '''return a boolean vector A, A[i] = True when it is unique.
        A segment is unique when the ratio between overlaping area over union area 
        between it and any other bigger segment is lower than the threshold.
       '''
    def __init__(self, threshold=0.9):
        self.threshold = threshold
    def _process( self, image, **args ):
        assert 'segments' in args
        segments = args['segments']
        if len(segments) == 0:
            return(image, args)
        
        segments = segments.astype(int)
        x1,y1= segments[:,0], segments[:,1]
        x2,y2= x1+segments[:,2], y1+segments[:,3]
        s = segments[:, 2] * segments[:, 3]
        n=len(segments)
        
        x1.shape = x2.shape = y1.shape = y2.shape = s.shape = (n, 1)
        X1 = numpy.repeat(x1, n, 1)
        X2 = numpy.repeat(x2, n, 1)
        Y1 = numpy.repeat(y1, n, 1)
        Y2 = numpy.repeat(y2, n, 1)
        S  = numpy.repeat(s , n, 1)

        
        #overlapping area
        SI= (numpy.maximum(0, numpy.minimum(X2, X2.T) - numpy.maximum(X1, X1.T) ) *
             numpy.maximum(0, numpy.minimum(Y2, Y2.T) - numpy.maximum(Y1, Y1.T)))
        #union area
        SU = S + S.T - SI
        #ratio area
        ratio = SI/SU.astype(float)
        #remove ratio which compare a segment with a bigger segment
        ratio[S>S.T] = 0.
        #deal with equal area segments
        ST0 = numpy.tril(S.T)
        ratio[S==ST0] = 0.
        ratio[ratio>=self.threshold] = 1
        uniques = numpy.logical_not (numpy.amax(ratio, axis = 1) == 1)
        return (image, merge_dicts(args, {'segments':segments[uniques]}))
