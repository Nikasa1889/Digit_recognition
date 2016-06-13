from utils import _same_type, show_image_and_wait_for_key, OverflowPreventer, merge_dicts
import numpy
import cv2
class Processor( object ):
    '''In goes something, out goes another. Processor.process() models 
    the behaviour of a function, where there are some stored parameters 
    in the Processor instance. Further, it optionally calls arbitrary 
    functions before and after processing (prehooks, posthooks)'''
    
    def _process( self, arguments ):
        raise NotImplementedError(str(self.__class__)+"."+"_process")
    
    def add_prehook( self, prehook_function ): #functions (on input) to be executed before processing
        if not hasattr(self, '_prehooks'):
            self._prehooks= []
        self._prehooks.append( prehook_function )
    
    def add_poshook( self, poshook_function ):
        if not hasattr(self, '_poshooks'):
            self._poshooks= []
        self._poshooks.append( poshook_function ) #functions (on output) to be executed after processing
     
    def process( self, image, **args):
        '''process function must return a 2-element tuple, 
        The first is output image, 
        The second is dictionary of other variable which could be used later in the stack
        '''
        self._input= merge_dicts({'image':image}, args)
        if hasattr(self, '_prehooks'):
            for prehook in self._prehooks:
                prehook( self )
        image_out, args_out= self._process(image, **args)
        self._output= merge_dicts({'image':image_out}, args_out)
        if hasattr(self, '_poshooks'):
            for poshook in self._poshooks:
                poshook( self )
        return (image_out, args_out)
    
    def get_output( self ):
        return self._output
    
    def get_input( self ):
        return self._input
    
    def display( self, display_before=True ):
        if display_before:
            show_image_and_wait_for_key(self._input['image'], "before "+self.__class__.__name__)
        show_image_and_wait_for_key(self._output['image'],  "after " +self.__class__.__name__)

class BrightnessProcessor( Processor ):
    '''changes image brightness. 
    A brightness of -1 will make the image all black; 
    one of 1 will make the image all white'''
    def __init__(self, brightness=0.0):
        assert -1<=brightness<=1
        self.brightness = brightness
    def _process( self , image, **args ):
        b= self.brightness
        assert image.dtype==numpy.int8
        image= image.copy()
        with OverflowPreventer(image) as img:
            img+=b*256
        return (image, args)
        
class BackgroundSubstractorGMGProcessor( Processor ):
    def __init__(self):
        self.bgsubstractor = cv2.createBackgroundSubtractorGMG()
    def _process( self, image, **args ):
        image = image.copy()
        image_mask = self.bgsubstractor(image)
        #TODO

class ContrastProcessor( Processor ):
    '''changes image contrast. a scale of 1 will make no changes'''
    def __init__( self, scale=1.0, center=0.5 ):
        self.scale = scale
        self.center = center
    def _process( self , image, **args ):
        assert image.dtype==numpy.uint8
        image= image.copy()
        s,c= self.scale, self.center
        c= int(c*256)
        with OverflowPreventer(image) as img:
            if s<=1:
                img*=s
                img+= int(c*(1-s))
            else:
                img-=c*(1 - 1/s)
                img*=s
        return (image, args)
class InvertProcessor( Processor ):
    '''Invert image color'''
    def _process( self , image, **args ):
        image = image.copy()
        image = (255-image)
        return (image, args)
    
class ScaleProcessor ( Processor ):
    '''Scale the image to a maximum height or width'''
    def __init__(self, maxHeight=800, maxWidth=800):
        self.maxHeight = maxHeight
        self.maxWidth = maxWidth
    
    def _process( self, image, **args ):
        image = image.copy()
        #if (image.shape[0] <= self.maxWidth) or (image.shape[1] <= self.maxHeight):
        #    return (image, args)
        scalex = self.maxWidth / float(image.shape[0])
        scaley = self.maxHeight / float(image.shape[1])
        scale = min(scalex, scaley)
        image = cv2.resize(image,(int(scale*image.shape[1]), int(scale*image.shape[0])), interpolation = cv2.INTER_LINEAR)
        return (image, args)

class SaveImageProcessor( Processor ):
    '''Save the input image into originalImg variable in the args dictionary of the pipeline'''
    def _process( self, image, **args ):
        return (image.copy(), merge_dicts(args, {'originalImg':image}))
    def display( self, display_before=True ):
        pass

class RetrieveImageProcessor( Processor ):
    '''Retrieve orignalImg from args to image pipeline'''
    def _process( self, image, **args ):
        return (args['originalImg'].copy(), args)
    def display( self, display_before=True ):
        pass

class HistogramEqualizationProcessor( Processor ):
    '''Equalize the histogram of image'''
    GLOBAL = 1
    CLAHE = 2
    def __init__( self, type = GLOBAL ):
        self.type = type
    
    def _process( self, image, **args ):
        image = image.copy()
        if self.type == self.GLOBAL:
            image = cv2.equalizeHist(image)
        elif self.type == self.CLAHE:
            image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image)
        
        return (image, args)

class HoughLinesProcessor( Processor ):
    '''HoughLine Probabilistic algorithm'''
    def _process( self, image, **args ):
        image = image.copy()
        lines = cv2.HoughLinesP(image,1,numpy.pi/180,100,minLineLength=20,maxLineGap=50)
        image = numpy.zeros(image.shape)
        for line in lines[0]:
            x1,y1,x2,y2 = line
            cv2.line(image,(x1,y1),(x2,y2),(255,255,0),thickness = 1)
        return (image, args)
    
class GrayScaleProcessor(Processor ):
    '''Convert image to grayscale, we assume color image does not help in this case'''
    def _process( self , image, **args ):
        assert image.dtype==numpy.uint8
        image= image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return (image, args)

class SobelProcessor( Processor ):
    '''Edge detector using Sobel algorithm'''
    def __init__( self, xorder=1, yorder=0, ksize=3, scale=1, ddepth=cv2.CV_8U ):
        self.xorder = xorder
        self.yorder = yorder
        self.ksize = ksize
        self.scale = scale
        self.ddepth = ddepth
        
    def _process( self, image, **args ):
        image = image.copy()
        image = cv2.Sobel(image, self.ddepth, self.xorder, self.yorder, self.ksize, self.scale)
        return (image, args)

class CannyProcessor( Processor ):
    '''Canny edge dectector algorithm'''
    def __init__(self, minVal = 100, maxVal = 200):
        self.minVal = minVal
        self.maxVal = maxVal
    
    def _process( self, image, **args ):
        image = image.copy()
        image = cv2.Canny(image,self.minVal, self.maxVal)
        return (image, args)
    
class ThresholdProcessor( Processor ):
    '''Apply a threshold to obtain a binary image'''
    OTSU= 1
    ADAPTIVE=2
    
    def __init__(self, type=OTSU, adaptiveBlockSize = 21):
        self.type = type
        self.blockSize = adaptiveBlockSize
    def _process( self, image, **args ):
        image= image.copy()
        if (self.type == self.OTSU):
            threshold, image = cv2.threshold( image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU )
        elif (self.type == self.ADAPTIVE):
            image = cv2.adaptiveThreshold( image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          thresholdType=cv2.THRESH_BINARY, blockSize=self.blockSize, C=0 )
            
        return (image, args)

    
class MorphologyProcessor( Processor ):
    '''Apply a Morphological operations'''
    CLOSE = 1
    OPEN = 2
    ERODE = 3
    DILATE = 4
    def __init__(self, shape = cv2.MORPH_RECT, ksize= (17,3), op = CLOSE):
        if (ksize == (0, 0)):
            self.morph_element = None
        else:
            self.morph_element = cv2.getStructuringElement(shape, ksize)
        self.op = op
        
    def _process(self, image, **args):
        image = image.copy()
        if (self.morph_element != None):
            if self.op == self.CLOSE:
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.morph_element)
            elif self.op == self.OPEN:
                image = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.morph_element)
            elif self.op == self.ERODE:
                image = cv2.erode(image, self.morph_element, 1)
            elif self.op == self.DILATE:
                image = cv2.dilate(image, self.morph_element, 1)
        return (image, args)
        
class BlurProcessor( Processor ):
    '''Blur to reduce noise'''
    def __init__( self, blur_x=0, blur_y=0 ):
        self.blur_x = blur_x
        self.blur_y = blur_y
    def _process( self , image, **args ):
        assert image.dtype==numpy.uint8
        image= image.copy()
        x, y= self.blur_x, self.blur_y
        x+= (x+1)%2 #opencv needs a
        y+= (y+1)%2 #odd number...
        image = cv2.GaussianBlur(image,(x,y),0)
        return (image, args)
        
