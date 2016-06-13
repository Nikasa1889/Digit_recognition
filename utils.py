import numpy
import math
import cv2
import functools
import sys
def _same_type(a,b):
    type_correct=False
    if type(a)==type(b):
        type_correct=True
    try: 
        if isinstance(a, b):
            type_correct=True
    except TypeError: #v may not be a class or type, but an int, a string, etc
        pass
    return type_correct
    
def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def ask_for_key( return_arrow_keys=True ):
    key=128
    while key > 127:
        key=cv2.waitKey(0)
        if return_arrow_keys:
            if key in (65362,65364,65361,65363): #up, down, left, right
                return key
        key %= 256
    return key

def background_color( image, numpy_result=True ):
    result= numpy.median(numpy.median(image, 0),0).astype( numpy.int )
    if not numpy_result:
        try:
            result= tuple(map(int, result))
        except TypeError:
            result= (int(result),)
    return result
    
def show_image_and_wait_for_key( image, name="Image" ):
    '''Shows an image, outputting name. keygroups is a dictionary of keycodes to functions; they are executed when the corresponding keycode is pressed'''
    print "showing",name,"(waiting for input)"
    sys.stdout.flush()
    
    cv2.imshow('norm',image)
    return ask_for_key()
        
def draw_segments( image , segments, color=(255,0,0), line_width=1):
        '''draws segments on image'''
        for segment in segments:
            x,y,w,h= segment
            cv2.rectangle(image,(x,y),(x+w,y+h),color,line_width)

def draw_lines( image, ys, color= (255,0,0), line_width=1):
    '''draws horizontal lines'''
    for y in ys:
        cv2.line( image, (0,y), (image.shape[1], y), color, line_width )

def draw_classes( image, segments, classes, scale = 0.5, thickness = 1):
    assert len(segments)==len(classes)
    for s,c in zip(segments, classes):
        x,y,w,h=s
        cv2.putText(image,str(c),(x,y),0,scale,(128,128,128), thickness=thickness)
        
def draw_classes_with_bg( image, segments, classes, scale = 0.5, thickness = 1, color_bg = (0,0,255)):
    assert len(segments)==len(classes)
    for s,c in zip(segments, classes):
        x,y,w,h=s
        ((textWidth, textHeight), baseline) = cv2.getTextSize(str(c), 0, scale, thickness)
        cv2.rectangle(image, (x, y-textHeight), (x+textWidth, y), color_bg, -1)
        cv2.putText(image,str(c),(x,y),0,scale,(255,255,255), thickness=thickness)
    
def draw_probs( image, segments, probs ):
    assert len(segments) == len(probs)
    for s, p in zip (segments, probs):
        x, y, w, h = s
        cv2.putText(image, str(round(p, 1)), (x, y+h), 0, 0.5, (255, 128, 128))
class OverflowPreventer( object ):
    '''A context manager that exposes a numpy array preventing simple operations from overflowing.
    Example:
    array= numpy.array( [255], dtype=numpy.uint8 )
    with OverflowPreventer( array ) as prevented:
        prevented+=1
    print array'''
    inverse_operator= { '__iadd__':'__sub__', '__isub__':'__add__', '__imul__': '__div__', '__idiv__':'__mul__'}
    bypass_operators=['__str__', '__repr__', '__getitem__']
    def __init__( self, matrix ):
        class CustomWrapper( object ):
            def __init__(self, matrix):
                assert matrix.dtype==numpy.uint8
                self.overflow_matrix= matrix
                self.overflow_lower_range= float(0)
                self.overflow_upper_range= float(2**8-1)
                for op in OverflowPreventer.bypass_operators:
                    setattr(CustomWrapper, op, getattr(self.overflow_matrix, op))
            
            def _overflow_operator( self, b, forward_operator):
                m, lr, ur= self.overflow_matrix, self.overflow_lower_range, self.overflow_upper_range
                assert type(b) in (int, float)
                reverse_operator= OverflowPreventer.inverse_operator[forward_operator]
                uro= getattr( ur, reverse_operator)
                lro= getattr( lr, reverse_operator)
                afo= getattr( m, forward_operator )
                overflows= m > uro( b )
                underflows= m < lro( b )
                afo(int(math.floor(b))) #Careful, very dangerous with mul and div operators
                m[overflows]= ur
                m[underflows]= lr
                return self
                
            def __getattr__(self, attr):
                if hasattr(self.wrapped, attr):
                    return getattr(self.wrapped,attr)
                else:
                    raise AttributeError

        self.wrapper= CustomWrapper(matrix)
        for op in OverflowPreventer.inverse_operator.keys():
            setattr( CustomWrapper, op, functools.partial(self.wrapper._overflow_operator, forward_operator=op))

    def __enter__( self ):
        return self.wrapper
    
    def __exit__( self, type, value, tb ):
        pass

def contained_segments_matrix( segments ):
    '''givens a n*n matrix m, n=len(segments), in which m[i,j] means
    segments[i] is contained inside segments[j]'''
    x1,y1= segments[:,0], segments[:,1]
    x2,y2= x1+segments[:,2], y1+segments[:,3]
    n=len(segments)
    
    x1so, x2so,y1so, y2so= map(numpy.argsort, (x1,x2,y1,y2))
    x1soi,x2soi, y1soi, y2soi= map(numpy.argsort, (x1so, x2so, y1so, y2so)) #inverse transformations
    o1= numpy.triu(numpy.ones( (n,n) ), k=1).astype(bool) # let rows be x1 and collumns be x2. this array represents where x1<x2
    o2= numpy.tril(numpy.ones( (n,n) ), k=0).astype(bool) # let rows be x1 and collumns be x2. this array represents where x1>x2
    
    a_inside_b_x= o2[x1soi][:,x1soi] * o1[x2soi][:,x2soi] #(x1[a]>x1[b] and x2[a]<x2[b])
    a_inside_b_y= o2[y1soi][:,y1soi] * o1[y2soi][:,y2soi] #(y1[a]>y1[b] and y2[a]<y2[b])
    a_inside_b= a_inside_b_x*a_inside_b_y
    return a_inside_b
