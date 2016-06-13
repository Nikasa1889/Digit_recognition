from processor import Processor, BrightnessProcessor
from utils import contained_segments_matrix, merge_dicts, draw_segments, show_image_and_wait_for_key, draw_classes, draw_probs
from tesseract_utils import segments_to_numpy

import cv2
import numpy
import theano
import theano.tensor as T

import lasagne


class SegmentClassifier( Processor ):
    '''Processor for segments, given image and segments, returning only the desirable
    ones'''
    COLOR =[(0, 0, 0), (255, 255, 255), 
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 150, 150),(150, 255, 150), (150, 150, 255)]
    def display( self, display_before=False):
        '''shows the effect of this filter'''
        try:
            image = self._input['originalImg'].copy()
        except KeyError:
            try:
                image = self._input['image'].copy()
            except KeyError:
                raise Exception("You need to set the _input['image'] or _input['originalImg'] attribute for displaying")
        segments = self._output['segments']
        segment_types = self._output['segment_types']
        unique_types = numpy.unique(segment_types)
        for segment_type in unique_types:
            #if (segment_type != 10):
            draw_segments( image, segments[segment_types == segment_type], self.COLOR[segment_type], line_width = 1)
        draw_classes( image, segments, segment_types)
        
        #if ('segment_types_prob' in self._output):
            #draw_probs( image, segments, self._output['segment_types_prob'] )
            
        show_image_and_wait_for_key( image, "segments classified by "+self.__class__.__name__)
        
class ManualSegmentClassifier( SegmentClassifier ):
    def _process( self, image, **args ):
        assert 'segments' in args
        assert 'regions' in args
        segments = args['segments']
        regions = args['regions']
        if 'segment_types' in args:
            segment_types = args['segment_types'].copy()
        else:
            segment_types = numpy.repeat(10, len(segments))
        
        self.x1,self.y1= segments[:,0], segments[:,1]
        self.x2,self.y2= self.x1+segments[:,2], self.y1+segments[:,3]
        self.s = segments[:, 2] * segments[:, 3]
        
        self._output = {'segments': segments, 'segment_types':segment_types}
        
        global refPt
        refPt = None
        # keep looping until the 'q' key is pressed
        cv2.namedWindow("norm", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("norm", self.click_handler)
        cv2.namedWindow("selected_region", cv2.WINDOW_NORMAL)
        classifying = True
        selected_segment = -1
        while classifying:
            # display the image and wait for a keypress
            self.display()
            key = cv2.waitKey(1) & 0xFF 
            # if the 'q' key is pressed, break from the loop
            if key == 27:
                classifying = False
                break
            
            if (refPt != None):
                selected_segment = self.get_selected_segment(refPt)
                refPt = None
            
            if (selected_segment >=0):
                cv2.imshow("selected_region", regions[selected_segment])
                char = chr(key)
                key = 0
                if (char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                    digit = int(char)
                    self.update_click_class(selected_segment, digit)
                    selected_segment = -1
                elif (char == 'r'):
                    digit = 10
                    self.update_click_class(selected_segment, digit)
                    selected_segment = -1
                
                
        cv2.destroyAllWindows()
        return (image, merge_dicts(args, {'segment_types':segment_types})) 
    
    def get_selected_segment(self, point):
        pointx, pointy = point
        selected_segments = numpy.where((self.x1 <= pointx) & (self.y1 <= pointy) & (self.x2 >= pointx) & (self.y2 >= pointy))[0]
        if len(selected_segments > 0):
            selected_segment = selected_segments[numpy.argmin(self.s[selected_segments])]
            return selected_segment
        else:
            return -1
            
    def update_click_class(self, selected_segment, digit):
        ####
        self._output['segment_types'][selected_segment] = digit
    
    def click_handler(self, event, x, y, flags, param):
        # grab references to the global variables
        global refPt
 
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = (x, y)


class NonDigitSegmentClassifier( SegmentClassifier ):
    
    def __init__(self, loose_high_threshold = 0.75, loose_low_threshold = 0.15, occupy_threshold = 0.75):
        self.loose_high_threshold = loose_high_threshold
        self.loose_low_threshold = loose_low_threshold
        self.occupy_threshold = occupy_threshold
        
    def _process( self, image, **args ):
        assert 'digit_segments' in args
        assert 'digit_segment_types' in args
        assert 'non_digit_segments' in args
        
        digit_segments = args['digit_segments']
        digit_segment_types = args['digit_segment_types']
        non_digit_segments = args['non_digit_segments']
        assert len(digit_segments) > 0
        if len(non_digit_segments)  == 0:
            return (image, merge_dicts(args, {'segments':digit_segments, 'segment_types':digit_segment_types}))
        
        digit_segments = digit_segments.astype(int)
        non_digit_segments = non_digit_segments.astype(int)
        
        x1,y1= non_digit_segments[:,0], non_digit_segments[:,1]
        x2,y2= x1+non_digit_segments[:,2], y1+non_digit_segments[:,3]
        
        x1_, y1_= digit_segments[:,0], digit_segments[:,1]
        x2_, y2_= x1_+digit_segments[:,2], y1_+digit_segments[:,3]
        
        n=len(non_digit_segments)
        m=len(digit_segments)
        
        x1.shape = x2.shape = y1.shape = y2.shape = (n, 1)
        x1_.shape = x2_.shape = y1_.shape = y2_.shape = (1, m)
        X1 = numpy.repeat(x1, m, 1)
        X2 = numpy.repeat(x2, m, 1)
        Y1 = numpy.repeat(y1, m, 1)
        Y2 = numpy.repeat(y2, m, 1)
        
        X1_ = numpy.repeat(x1_, n, 0)
        X2_ = numpy.repeat(x2_, n, 0)
        Y1_ = numpy.repeat(y1_, n, 0)
        Y2_ = numpy.repeat(y2_, n, 0)
        
        
        
        loose_X = (numpy.maximum(X1-X1_, 0) + numpy.maximum(X2_-X2, 0))/(X2_-X1_).astype(float)
        loose_Y = (numpy.maximum(Y1-Y1_, 0) + numpy.maximum(Y2_-Y2, 0))/(Y2_-Y1_).astype(float)
        
        occupy_X = (numpy.minimum(X2, X2_)-numpy.maximum(X1, X1_))/(X2-X1).astype(float)
        occupy_Y = (numpy.minimum(Y2, Y2_)-numpy.maximum(Y1, Y1_))/(Y2-Y1).astype(float)
        
        loose = numpy.maximum(loose_X, loose_Y)
        occupy = numpy.minimum(occupy_X, occupy_Y)
        
        non_digit_segment_types = numpy.empty(len(non_digit_segments), dtype = int)
        non_digit_segment_types.fill(10)
        
        isRemoved = (loose <= self.loose_high_threshold) & (loose >= self.loose_low_threshold)
        isRemoved = isRemoved | ((loose<=self.loose_low_threshold) & (occupy<= self.occupy_threshold))
        isRemoved = numpy.any(isRemoved, axis = 1)
        
        isCovered = (loose <= self.loose_low_threshold) & (occupy >= self.occupy_threshold)
        
        isCovered = numpy.any(isCovered, axis = 1)
        covered_digits = numpy.argmax(occupy, axis = 1)
        #coveredPairs = numpy.nonzero(isCovered)
        #non_digit_segment_types = numpy.empty(len(non_digit_segments), dtype = int)
        #non_digit_segment_types.fill(10)
        non_digit_segment_types = digit_segment_types[covered_digits]
        non_digit_segment_types[~isCovered] = 10
        
        non_digit_segments = non_digit_segments[~isRemoved]
        non_digit_segment_types = non_digit_segment_types[~isRemoved]
        #non_digit_segment_types[coveredPairs[0]] = digit_segment_types[coveredPairs[1]]
        
        segments = numpy.concatenate((digit_segments, non_digit_segments), axis = 0)
        segment_types = numpy.concatenate((digit_segment_types, non_digit_segment_types), axis = 0)
        
        return (image, merge_dicts(args, {'segments':segments, 'segment_types':segment_types}))
        

class DeepCnnSegmentClassifier( SegmentClassifier ):
    def build_cnn_deep(self, input_var=None, n_output = 10):
        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                            input_var=input_var)
        #conv1
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=64, filter_size=(3, 3), pad = 1,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        #conv2
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=64, filter_size=(3, 3), pad = 1,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        #max2
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)
        #dropout2
        network = lasagne.layers.dropout(network, p=.25)

        #conv3
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=128, filter_size=(3, 3), pad = 1,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        #conv4
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=128, filter_size=(3, 3), pad = 1,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        #max4
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2)
        #dropout4
        network = lasagne.layers.dropout(network, p=.25)

        #conv5
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=256, filter_size=(3, 3), pad = 1,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        #conv6
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=256, filter_size=(3, 3), pad = 1,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        #conv7
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=256, filter_size=(3, 3), pad = 1,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        #conv8
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=256, filter_size=(3, 3), pad = 1,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        #max8
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)
        #dropout8
        network = lasagne.layers.dropout(network, p=.25)

        #full1
        network = lasagne.layers.DenseLayer(
                network,
                num_units=1024,
                nonlinearity=lasagne.nonlinearities.rectify)

        network = lasagne.layers.dropout(network, p=.5)

        #full2
        network = lasagne.layers.DenseLayer(
                network,
                num_units=1024,
                nonlinearity=lasagne.nonlinearities.rectify)

        network = lasagne.layers.dropout(network, p=.5)


        # And, finally, the 10-unit output layer with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
                network,
                num_units=n_output,
                nonlinearity=lasagne.nonlinearities.softmax)
        return network
    
    def __init__(self, trained_network_file, n_output = 10):
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        self.network = self.build_cnn_deep(input_var, n_output)
        with numpy.load(trained_network_file) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)

        prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.predict_fn = theano.function([input_var], prediction, mode = 'FAST_RUN')
        
    
    def _process( self, image, **args ):
        assert 'segments' in args
        assert 'regions' in args
        segments = args['segments']
        regions = args['regions']
        #regions produced by feature_extractor has different dimension setting to lagsane network
        #adapt it here
        regions = numpy.rollaxis(regions, 3, 1)
        predictions = self.predict_fn(regions)
        segment_types = numpy.argmax(predictions, axis = 1)
        segment_types_prob = numpy.amax(predictions, axis = 1)
        
        return (image, merge_dicts(args, {'segment_types':segment_types, 'segment_types_prob':segment_types_prob}))

class DeepBNCnnSegmentClassifier( SegmentClassifier ):
    def build_cnn_deep(self, input_var=None, n_output = 11):
        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                            input_var=input_var)
        # This time we do not apply input dropout, as it tends to work less well
        # for convolutional layers.

        #conv1
        network = lasagne.layers.normalization.batch_norm(
                    lasagne.layers.Conv2DLayer(
                    network, num_filters=64, filter_size=(3, 3), pad = 1,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform()),
                    alpha = 0.005)
        #conv2
        network = lasagne.layers.Conv2DLayer(
                    network, num_filters=64, filter_size=(3, 3), pad = 1,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform())
        #max2
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)
        #dropout2
        network = lasagne.layers.dropout(network, p=.25)

        #conv3
        network = lasagne.layers.batch_norm(
                    lasagne.layers.Conv2DLayer(
                    network, num_filters=128, filter_size=(3, 3), pad = 1,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform()),
                    alpha = 0.005
                    )
        #conv4
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=128, filter_size=(3, 3), pad = 1,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        #max4
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2)
        #dropout4
        network = lasagne.layers.dropout(network, p=.25)

        #conv5
        network = lasagne.layers.batch_norm(
                    lasagne.layers.Conv2DLayer(
                    network, num_filters=256, filter_size=(3, 3), pad = 1,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform()),
                    alpha = 0.005
                )
        #conv6
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=256, filter_size=(3, 3), pad = 1,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        #conv7
        network = lasagne.layers.batch_norm(
                    lasagne.layers.Conv2DLayer(
                    network, num_filters=256, filter_size=(3, 3), pad = 1,
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.GlorotUniform()),
                    alpha = 0.005
                )
        #conv8
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=256, filter_size=(3, 3), pad = 1,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        #max8
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)
        #dropout8
        network = lasagne.layers.dropout(network, p=.25)

        #full1
        network = lasagne.layers.batch_norm(
                    lasagne.layers.DenseLayer(
                    network,
                    num_units=1024,
                    nonlinearity=lasagne.nonlinearities.rectify),
                    alpha = 0.005
                )

        network = lasagne.layers.dropout(network, p=.5)

        #full2
        network = lasagne.layers.batch_norm(
                    lasagne.layers.DenseLayer(
                    network,
                    num_units=1024,
                    nonlinearity=lasagne.nonlinearities.rectify),
                    alpha = 0.005
                  )

        network = lasagne.layers.dropout(network, p=.5)


        # And, finally, the 10-unit output layer with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
                network,
                num_units=11,
                nonlinearity=lasagne.nonlinearities.softmax)

        return network

    
    def __init__(self, trained_network_file, n_output = 11):
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        self.network = self.build_cnn_deep(input_var, n_output)
        with numpy.load(trained_network_file) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)

        prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.predict_fn = theano.function([input_var], prediction, mode = 'FAST_RUN')
        
    
    def _process( self, image, **args ):
        assert 'segments' in args
        assert 'regions' in args
        segments = args['segments']
        regions = args['regions']
        #regions produced by feature_extractor has different dimension setting to lagsane network
        #adapt it here
        regions = numpy.rollaxis(regions, 3, 1)
        predictions = self.predict_fn(regions)
        segment_types = numpy.argmax(predictions, axis = 1)
        segment_types_prob = numpy.amax(predictions, axis = 1)
        
        return (image, merge_dicts(args, {'segment_types':segment_types, 'segment_types_prob':segment_types_prob}))

class DeepCnnSigmoidSegmentClassifier( DeepCnnSegmentClassifier ):
    def build_cnn_deep(self, input_var=None, n_output = 10):
        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                            input_var=input_var)
        #conv1
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3), pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        #conv2
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3), pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        #max2
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)
        #dropout2
        network = lasagne.layers.dropout(network, p=.25)

        #conv3
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3), pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        #conv4
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3), pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        #max4
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2)
        #dropout4
        network = lasagne.layers.dropout(network, p=.25)

        #conv5
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3), pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        #conv6
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3), pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        #conv7
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3), pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        #conv8
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(3, 3), pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        #max8
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride = 2)
        #dropout8
        network = lasagne.layers.dropout(network, p=.25)

        #full1
        network = lasagne.layers.DenseLayer(
            network,
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify)

        network = lasagne.layers.dropout(network, p=.5)

        #full2
        network = lasagne.layers.DenseLayer(
            network,
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify)

        network = lasagne.layers.dropout(network, p=.5)


        # And, finally, the 10-unit output layer with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
            network,
            num_units=n_output,
            nonlinearity=lasagne.nonlinearities.sigmoid)
        return network

class CnnSegmentClassifier( SegmentClassifier ):
    def build_cnn(self, input_var=None, n_output = 2):
        network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                            input_var=input_var)
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        network = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=256,
                nonlinearity=lasagne.nonlinearities.rectify)

        network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.5),
                num_units=n_output,
                nonlinearity=lasagne.nonlinearities.softmax)

        return network
    def __init__(self, trained_network_file, n_output = 2):
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        self.network = self.build_cnn(input_var, n_output)
        with numpy.load(trained_network_file) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)

        prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.predict_fn = theano.function([input_var], prediction)
        
    
    def _process( self, image, **args ):
        assert 'segments' in args
        assert 'regions' in args
        segments = args['segments']
        regions = args['regions']
        #regions produced by feature_extractor has different dimension setting to lagsane network
        #adapt it here
        regions = numpy.rollaxis(regions, 3, 1)
        segment_types = numpy.argmax(self.predict_fn(regions), axis = 1)
        return (image, merge_dicts(args, {'segment_types':segment_types}))