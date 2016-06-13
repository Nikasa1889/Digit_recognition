import cv2
import numpy
import theano
import theano.tensor as T

import lasagne

from processor import Processor
from utils import merge_dicts, draw_segments, show_image_and_wait_for_key, draw_classes_with_bg

class PowerNumberReader ( Processor ):
    def __init__( self, bank_extension = 0, bank_cut_off_ratio = 0.8, 
                 height_error_acceptance = 0.3, distance_error_acceptance = 0.4, residual_threshold = 3):
        self.bank_extension = bank_extension
        self.bank_cut_off_ratio = bank_cut_off_ratio
        self.height_error_acceptance = height_error_acceptance
        self.distance_error_acceptance = distance_error_acceptance
        self.residual_threshold = residual_threshold
    COLOR =[(0, 0, 0), (255, 255, 255), 
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 150, 150),(150, 255, 150), (150, 150, 255)]
    
    def draw_result( self, sequences, banks ):
        '''shows the effect of this filter'''
        try:
            image = self._input['originalImg'].copy()
        except KeyError:
            try:
                image = self._input['image'].copy()
            except KeyError:
                raise Exception("You need to set the _input['image'] or _input['originalImg'] attribute for displaying")

        i = 1
        for (sequence_segments, sequence_types) in sequences:
            color = self.COLOR[i%11]
            i = i+1
            draw_segments( image, sequence_segments, color, line_width = 2)
            draw_classes_with_bg( image, sequence_segments, sequence_types, scale=0.8, thickness=2)
        
        i = 1
        for bank in banks:
            color = self.COLOR[i%11]
            i = i + 1
            cv2.line(image,(0,bank['miny']),(image.shape[1],bank['miny']),color,2)
            cv2.line(image,(0,bank['maxy']),(image.shape[1],bank['maxy']),color,2)
        
        return image;
    
    def  display( self, display_before=False):
        show_image_and_wait_for_key( self._output["image"], "segments filtered by "+self.__class__.__name__)
        
        
    def _process( self, image, **args ):
        assert 'segments' in args
        assert 'regions' in args
        assert 'segment_types' in args
        segments = args['segments']
        segment_types = args['segment_types']
        regions = args['regions']
        banks = self.find_banks( segments, segment_types )
        print banks
        sequences = []
        good_banks = []
        for bank in banks:
            (sequence_segments, sequence_types) = self.find_best_digit_sequence( bank, segments, segment_types )
            if (len(sequence_segments) > 0 and self.is_sequence_on_line( sequence_segments, self.residual_threshold)):
                sequences.append((sequence_segments, sequence_types) )
                good_banks.append(bank)
        image = self.draw_result(sequences, good_banks)
        return (image, merge_dicts(args, {'sequences':sequences, 'banks': good_banks}))
            
    def is_sequence_on_line( self, segments, residual_threshold ):
        xs = []
        ys = []
        height = 0;
        for segment in segments:
            x, y, w, h = segment
            height = height + h
            xs.append(x)
            ys.append(y)
        height = height / len(segments)
        xs = numpy.asarray(xs)
        ys = numpy.asarray(ys)
        _, residuals, _, _, _ = numpy.polyfit(xs, ys, deg=1, full=True)
        
        residual = numpy.sqrt(residuals.mean())/len(segments)
        residual = residual/height
        print 'Residual: ', residual
        if (residual<=residual_threshold):
            return True
        else:
            return False
        
    def find_banks ( self, segments, segment_types ):
        digit_ids = numpy.where(segment_types[:]!= 10)
        banks = []
        for digit_id in numpy.nditer(digit_ids):
            banks.append(self.find_bank(digit_id, segments))
        #Bug, can't use this filter
        #banks = self.remove_too_big_center_digit (banks, segments, segment_types)
        banks = self.remove_redundant_banks(banks, segment_types)
        banks = self.keep_largest_banks(banks)
        return banks
        
    def find_bank (self, digit_id, segments ):
        center_segment = segments[digit_id]
        x, y, w, h = center_segment
        miny = y-self.bank_extension
        maxy = y+h+self.bank_extension
        segment_ids = self.find_segment_inside_bank (miny, maxy, segments)
        return {'center_digit_id':digit_id, 'segment_ids':segment_ids, 'miny': miny, 'maxy':maxy}
    
    def find_segment_inside_bank ( self, miny, maxy, segments ):
        y1, y2, h = segments[:, 1], segments[:, 1]+segments[:, 3], segments[:, 3]
        bank_h = maxy - miny
        #TODO: doing this might miss rotated numbers
        #mid_y = (y1+y2)/2
        #not_inside_segments=  (mid_y  <= miny) | (mid_y >= maxy)
        not_inside_segments=  (y2  <= miny) | (y1 >= maxy)
        inside_segments = numpy.logical_not(not_inside_segments) & (numpy.absolute(h-bank_h)<0.7*bank_h)
        inside_segment_ids = numpy.where(inside_segments)
        
        return inside_segment_ids
    

    #Bug here!
    def remove_too_big_center_digit (self, banks, segments, segment_types):
        banks = sorted(banks, key=lambda bank: bank['maxy']-bank['miny'], reverse=True)
        kept_banks = []
        for bank in banks:
            bank_digit_ids = bank['segment_ids'][0][segment_types[bank['segment_ids']]!=10]
            if (len(bank_digit_ids)<3):
                continue
            bank_digit_segments = segments[bank_digit_ids]            
            bank_digit_segments = sorted(bank_digit_segments, key = lambda segment: segment[3], reverse=True)
            if (bank_digit_segments[0][3] <= 1.5*bank_digit_segments[1][3]):
                kept_banks.append(bank)
            
        return kept_banks
            
    def remove_redundant_banks( self, banks, segment_types ):
        banks = sorted(banks, key=lambda bank: numpy.count_nonzero(segment_types[bank['segment_ids']]!=10), reverse=True)
        kept_banks = []
        appeared = numpy.empty(shape = (len(segment_types), ), dtype = bool)
        appeared.fill(False)
        for bank in banks:
            n_digits = numpy.count_nonzero(segment_types[bank['segment_ids']]!=10)
            if (n_digits <= 3):
                continue
            if not appeared[bank['center_digit_id']]:
                kept_banks.append(bank)
                appeared[bank['segment_ids']] = True
        return kept_banks
    
    def keep_largest_banks( self, banks ):
        banks = sorted(banks, key=lambda bank: bank['maxy']-bank['miny'], reverse=True)
        kept_banks = []
        height = None
        for bank in banks:
            bank_h = bank['maxy'] - bank['miny']
            if (height == None):
                height = bank_h
                kept_banks.append(bank)
            else:
                ref_height = height/float(len(kept_banks))
                if ((bank_h/ref_height)>=self.bank_cut_off_ratio):
                    height = height + bank_h
                    kept_banks.append(bank)
                else:
                    return kept_banks
        return kept_banks
   
    def find_best_digit_sequence( self, bank, segments, segment_types ):
        bank_segments = segments[bank['segment_ids']]
        bank_segment_types = segment_types[bank['segment_ids']]
        
        sorted_idx = sorted(range(len(bank_segments)), key = lambda k: bank_segments[k, 0])
        bank_segments = bank_segments[sorted_idx, :]
        bank_segment_types = bank_segment_types[sorted_idx]
        sequences = []
        for start_idx in range(len(bank_segments)):
            #Always start with a digit
            if (bank_segment_types[start_idx] != 10):
                sequences.extend(self.find_sequence(start_idx, 
                                                         [bank_segments[start_idx, 3]], 
                                                         [], 
                                                         [start_idx], 
                                                         bank_segments, 
                                                         bank_segment_types))
        #Find the best sequence in sequences. The best is the one that goes through many recognized digits, 
        #     and has smallest standard deviation in distance and height 
        sequences = sorted(sequences, key = lambda sequence: (sequence['n_digits'], -len(sequence['segments']),
                                                              2 - sequence['std_distance']-sequence['std_height']), reverse=True)
        if len(sequences) == 0:
            return ([], [])
        else:
            return (bank_segments[sequences[0]['segments'], :], bank_segment_types[sequences[0]['segments']])
        
    def find_sequence( self, current_idx, heights, distances, added_segments, bank_segments, bank_segment_types ):
        
        ref_height =  numpy.mean(numpy.asarray(heights))
        if len(distances) == 0:
            ref_distance = 0
        else:
            ref_distance = numpy.mean(numpy.asarray(distances))
            if (ref_distance > 2*ref_height):
                return []
            
        if len(added_segments) >= 4:
            n_digits = numpy.count_nonzero(bank_segment_types[numpy.asarray(added_segments)]!=10)
            if (n_digits >=3):
                mean_distance = ref_distance
                mean_height = ref_height
                std_distance = numpy.std(numpy.asarray(distances))
                std_height = numpy.std(numpy.asarray(heights))
                sequences = [{'mean_distance':mean_distance,
                              'mean_height': mean_height,
                              'std_distance': std_distance,
                              'std_height': std_height,
                              'n_digits': n_digits,
                              'segments': added_segments}]
            else:
                sequences = []
        else:
            sequences = []

        if len(added_segments) <=6:
            for next_idx in range(current_idx+1, len(bank_segments)):
                if ((bank_segment_types[current_idx] != 10) and# or
                    (bank_segment_types[next_idx] != 10)): #or 
                    #(len(added_segments)>=3 and (bank_segment_types[added_segments[-3]])!= 10)):
                    distance = (bank_segments[next_idx, 0] + bank_segments[next_idx, 2]/2 
                                - bank_segments[current_idx, 0] - bank_segments[current_idx, 2]/2)
                    height = bank_segments[next_idx, 3]

                    if (bank_segments[next_idx, 0] - bank_segments[current_idx, 0] - bank_segments[current_idx, 2] 
                        <= -bank_segments[next_idx, 2]/3):
                        continue                

                    if ((len(distances) == 0) or   
                        (abs(height-ref_height) < self.height_error_acceptance*ref_height and 
                         abs(distance-ref_distance) < self.distance_error_acceptance * ref_distance)
                       ):
                        new_distances = list(distances)
                        new_distances.append(distance)
                        new_heights = list(heights)
                        new_heights.append(height)
                        new_added_segments = list(added_segments)
                        new_added_segments.append(next_idx)

                        new_sequences = self.find_sequence( next_idx, 
                                                      new_heights,
                                                      new_distances,
                                                      new_added_segments,
                                                      bank_segments,
                                                      bank_segment_types)                
                        sequences.extend(new_sequences)
        return sequences                                                        