import lmdb
import time
import numpy as np
import random
import caffe_pb2

class read_lmdb(object):

    def __init__(self, batch_size, lmdb_file, image_mean_path):
        self.lmdb_data = lmdb.open(lmdb_file,readonly=True)
        self.txn = self.lmdb_data.begin()
        self.lmdb_cursor = self.txn.cursor()
            #usage: init image shape 
        self.lmdb_cursor.first()
        (key, value) = self.lmdb_cursor.item()      
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        data = datum_to_array(datum)
        self.image_shape = data.shape
        #print "image shape:",self.image_shape
           # image_channels = data.shape[0]
           # image_height = data.shape[1]
           # image_width = data.shape[2]
            
	if image_mean_path is None:
	    self.image_mean = get_image_mean((104, 117, 123), self.image_shape)
	else:
	    self.image_mean = np.load(image_mean_path)
	
	self.total_number = int(self.lmdb_data.stat()['entries'])
	self.batch_size = batch_size
        self.total_batch = self.total_number / batch_size
	self.output_data = np.zeros((self.batch_size,)+self.image_shape, dtype='float32')
	self.output_label = np.zeros((self.batch_size,), dtype='int32')

    def __del__(self):
        #self.txn.close()
        self.lmdb_data.close()
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
	for i in xrange(self.batch_size):
	    (key, value) = self.lmdb_cursor.item()
            if self.lmdb_cursor.next() is False:
                self.lmdb_cursor.first()
                print "First....."
	    datum = caffe_pb2.Datum()
	    datum.ParseFromString(value)
            label = datum.label
            data = datum_to_array(datum)[np.newaxis].astype(np.float32)
            data = data - self.image_mean
            #data = datum_to_array(datum)[np.newaxis] - self.image_mean
	    self.output_data[i,...] = data
	    self.output_label[i] = label
        return (transform(self.output_data, self.image_shape), self.output_label) 

    def set_cursor(self, n):
        self.lmdb_cursor.first()
        if n > 0:
            for i in xrange(n*self.batch_size):
                self.lmdb_cursor.next()

def get_image_mean(image_mean, image_shape):
    if isinstance(image_mean, tuple):
	image_mean_out = np.zeros(image_shape)
	for i in xrange(len(image_mean)):
	    image_mean_out[i,...] = image_mean[i]
	image_mean_out = image_mean_out[np.newaxis]
    return image_mean_out

def transform(input, input_shape, crop_size=227, is_mirror=True):
    mirror = input[:, :, :, ::-1]
    crop_x = (input_shape[-1] - crop_size) / 2
    crop_y = (input_shape[-2] - crop_size) / 2
    #do_mirror = is_mirror and random.randint(0, 1)
    return input[:, :, crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
#    if False:
#	return mirror[:, :, crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
#    else:
#	return input[:, :, crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

def datum_to_array(datum):
    """Converts a datum to an array. Note that the label is not returned,
    as one can easily get it by calling datum.label.
    """
    if len(datum.data):
        return np.fromstring(datum.data, dtype=np.uint8).reshape(
            datum.channels, datum.height, datum.width)
    else:
        return np.array(datum.float_data).astype(float).reshape(
            datum.channels, datum.height, datum.width)

