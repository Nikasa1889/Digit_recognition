import numpy

CLASS_DATATYPE=     numpy.uint16
CLASS_SIZE=         1
CLASSES_DIRECTION=  0 #vertical - a classes COLUMN

BLANK_CLASS=        chr(35) #marks unclassified elements

def classes_to_numpy( classes ):
    '''given a list of unicode chars, transforms it into a numpy array'''
    import array
    #utf-32 starts with constant ''\xff\xfe\x00\x00', then has little endian 32 bits chars
    #this assumes little endian architecture!
    assert unichr(15).encode('utf-32')=='\xff\xfe\x00\x00\x0f\x00\x00\x00'
    assert array.array("I").itemsize==4
    int_classes= array.array( "I", "".join(classes).encode('utf-32')[4:])
    assert len(int_classes) == len(classes)
    classes=  numpy.array( int_classes,  dtype=CLASS_DATATYPE, ndmin=2) #each class in a column. numpy is strange :(
    classes= classes if CLASSES_DIRECTION==1 else numpy.transpose(classes)
    return classes

def classes_from_numpy(classes):
    '''reverses classes_to_numpy'''
    classes= classes if CLASSES_DIRECTION==0 else classes.tranpose()
    classes= map(unichr, classes)
    return classes

SEGMENT_DATATYPE=   numpy.uint16
SEGMENT_SIZE=       4
SEGMENTS_DIRECTION= 0 # vertical axis in numpy

def segments_from_numpy( segments ):
    '''reverses segments_to_numpy'''
    segments= segments if SEGMENTS_DIRECTION==0 else segments.tranpose()
    segments= [map(int,s) for s in segments]
    return segments

def segments_to_numpy( segments ):
    '''given a list of 4-element tuples, transforms it into a numpy array'''
    segments= numpy.array( segments, dtype=SEGMENT_DATATYPE, ndmin=2)   #each segment in a row
    segments= segments if SEGMENTS_DIRECTION==0 else numpy.transpose(segments)
    return segments

def region_from_segment( image, segment, extendTo = None ):
    '''given a segment (rectangle) and an image, returns it's corresponding subimage'''
    x,y,w,h= segment
    if extendTo != None:
        raise NotImplemented #must fix border!
    return image[y:y+h,x:x+w]


def read_boxfile( path ):
    classes=  []
    segments= []
    with open(path) as f:
        for line in f:
            s= line.split(" ")
            assert len(s)==6
            assert s[5]=='0\n'
            classes.append( s[0].decode('utf-8') )
            segments.append( map(int, s[1:5]))
    return classes_to_numpy(classes), segments_to_numpy(segments)

def write_boxfile(path, classes, segments):
    classes, segments= classes_from_numpy(classes), segments_from_numpy(segments)
    with open(path, 'w') as f:
        for c,s in zip(classes, segments):
            f.write( c.encode('utf-8')+' '+ ' '.join(map(str, s))+" 0\n")
