import numpy
import cv2

from processor import Processor
from utils import ask_for_key

class ProcessorStack (Processor):
    '''a stack of processors. Each processor's output is fed to the next'''
    def __init__(self, processor_instances=[]):
        self.set_processor_stack( processor_instances )

    def set_processor_stack( self, processor_instances ):
        assert all( isinstance(x, Processor) for x in processor_instances )
        self.processors= processor_instances

    def _process( self, image, **args ):
        for p in self.processors:
            image, args= p.process( image, **args )
        return (image, args)
        
    def display(self, display_before=False):
        pr= self.processors
        for p in pr:
            if hasattr(p, "display"):
                p.display( display_before )
        print 'Finish displaying ProcessorStack result, waiting for input'
        ask_for_key()
        cv2.destroyAllWindows()
                
class PlateDetector( ProcessorStack ):
    def __init__(self, stack):
        print stack
        ProcessorStack.__init__(self, stack)