import numpy
import cv2
from utils import background_color, merge_dicts, show_image_and_wait_for_key
from tesseract_utils import region_from_segment
from processor import Processor
FEATURE_DATATYPE=   numpy.uint8
#FEATURE_SIZE is defined on the specific feature extractor instance

class SimpleFeatureExtractor( Processor ):
    PAD_BACKGROUND = 1
    EXTEND_BOX = 2
    STRETCH = 3
    
    def __init__(self, feature_size=32, method=PAD_BACKGROUND):
        self.feature_size= feature_size
        self.method= method

    def _process(self, image, **args):
        try:
            segments = args['segments']
        except AttributeError:
            raise Exception("Can't find segments in args")
        image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        fs= self.feature_size
        bg= background_color( image )
 
        regions=numpy.ndarray( shape=(0,fs**2), dtype=FEATURE_DATATYPE )
        for segment in segments:
            if self.method == self.EXTEND_BOX:
                region= region_from_segment( image, segment, extendTo = self.feature_size )
            else:
                region= region_from_segment( image, segment, extendTo = None )
            if self.method == self.EXTEND_BOX or self.method == self.STRETCH:
                region = cv2.resize(region, (fs,fs) )
            elif self.method == self.PAD_BACKGROUND:
                x,y,w,h= segment
                proportion= float(min(h,w))/max(w,h)
                new_size= (fs, int(fs*proportion)) if min(w,h)==h else (int(fs*proportion), fs)
                region = cv2.resize(region, new_size)
                s= region.shape
                newregion= numpy.ndarray( (fs,fs), dtype= region.dtype )
                newregion[:,:]= bg
                startx, starty = int((fs-s[0])/2), int((fs-s[1]))/2 #center the region
                newregion[startx:(startx+s[0]),starty:(starty+s[1])]= region
                region=newregion
            region = region.reshape(1, fs**2)
            regions= numpy.append( regions, region, axis=0 )
        return (image, merge_dicts(args, {'regions':regions}))
    
    def display(self, display_before=True):
        '''Show all extracted region'''
        try:
            regions = self._output['regions']
        except AttributeError:
            raise Exception("You need to set the _output['regions'] attribute for displaying")
        if 'segment_types' in self._output:
            segment_types = self.output['segment_types']
        else:
            segment_types = numpy.array(len(regions), dtype = int)
            segment_types.fill(10)
        for (region, segment_type) in zip(regions, segment_types):
            x = region.reshape(self.feature_size, self.feature_size, 1)
            show_image_and_wait_for_key( x , "region" + str(segment_type))
            
class ThreeChannelsFeatureExtractor( Processor ):
    PAD_BACKGROUND = 1
    EXTEND_BOX = 2
    STRETCH = 3    
    def __init__(self, feature_size=32, extension = 0, method=PAD_BACKGROUND, display_features = False):
        self.feature_size= feature_size
        self.extension = extension
        self.display_features = display_features
        self.method = method
        print("Careful! PAD_BACKGROUND extension is now treated as percentage of dimension")

    def _process(self, image, **args):
        try:
            segments = args['segments']
        except AttributeError:
            raise Exception("Can't find segments in args")
        fs= self.feature_size
        extension = self.extension
        method = self.method
        image_h, image_w, _ = image.shape
        bg= background_color( image )
        regions=numpy.ndarray( shape=(0, fs, fs, 3), dtype=FEATURE_DATATYPE )
        
        if (method == self.EXTEND_BOX):
            #Pad the image to deal with regions at borders
            pad = (numpy.absolute(segments[:, 2]-segments[:, 3]).max())/2 + extension
            padded_img = cv2.copyMakeBorder(image,pad,pad+1,pad,pad+1,cv2.BORDER_CONSTANT,value=bg)
            segments = segments.astype(int)
            for segment in segments:
                x,y,w,h= segment
                extend_x = extend_y = 0
                if (w < h):
                    extend_x = h-w
                if (h < w):
                    extend_y = w-h
                new_segment = [x+pad-extend_x/2-extension, y+pad-extend_y/2-extension, w+extend_x+extension, h+extend_y+extension]
                region = region_from_segment( padded_img, new_segment )
                region = cv2.resize(region, (fs,fs))
                #show_image_and_wait_for_key( region, "test_region")
                region.shape = (1, fs, fs, 3)
                regions= numpy.append( regions, region, axis=0 )
        else:
            for segment in segments:
                x,y,w,h= segment
                
                x, y, w, h = max(0, x-extension), max(0, y-extension), min(w+extension, image_w-x), min(h+extension, image_h-y)
                segment = (x, y, w, h)
                pad_x = pad_y = 0
                if (w < h):
                    pad_x = (h-w)/2
                if (h < w):
                    pad_y = (w-h)/2
                region = region_from_segment( image, segment )
                #bg = background_color( region ) #Becareful
                newregion= numpy.ndarray( (max(w, h), max(w, h), 3), dtype= region.dtype )
                newregion[:,:]= bg
                newregion[pad_y:(pad_y+h), pad_x:(pad_x+w)] = region
                region = cv2.resize(newregion, (fs, fs))
                region.shape = (1, fs, fs, 3)
                regions= numpy.append( regions, region, axis = 0 )
            
        return (image, merge_dicts(args, {'regions':regions}))
    
    def display(self, display_before=True):
        '''Show all extracted region'''
        if self.display_features:
            try:
                regions = self._output['regions']
            except AttributeError:
                raise Exception("You need to set the _output['regions'] attribute for displaying")
            
            if 'segment_types' in self._output:
                segment_types = self._output['segment_types']
            else:
                segment_types = numpy.array(len(regions), dtype = int)
                segment_types.fill(10)
            for (region, segment_type) in zip(regions, segment_types):
                if segment_type != 10:
                #x = region.reshape(self.feature_size, self.feature_size, 1)
                    show_image_and_wait_for_key( region , "region" + str(segment_type))