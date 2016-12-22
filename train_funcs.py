import glob
import time
import os

import numpy as np

import hickle as hkl
#from my_read_lmdb import read_lmdb
from new_lmdb import *
#from proc_load import crop_and_mirror

#add by xiaotian
# add by owen
data_list = []

perf = [0.0,0.0,0.0]

def proc_configs(config):
    if not os.path.exists(config['weights_dir']):
        os.makedirs(config['weights_dir'])
        print "Creat folder: " + config['weights_dir']
    return config


def unpack_configs(config):
    print "-------------------------------------------"
    print "weight dir:"+str(config['weights_dir'])
    print "momentum:"+str(config['momentum'])
    print "weight_decay:"+str(config['weight_decay'])
    print "lr_step:"+str(config['lr_adapt_threshold'])
    print "-------------------------------------------"
    
    flag_para_load = config['para_load']
    train_lmdb_path = config['train_lmdb_path']
    val_lmdb_path = config['val_lmdb_path']
    image_mean_path = config['image_mean_path']
    return (flag_para_load, train_lmdb_path, val_lmdb_path, image_mean_path)

def adjust_learning_rate(config, epoch, step_idx, val_record, learning_rate):
    # Adapt Learning Rate
    if config['lr_policy'] == 'step':
        if epoch == config['lr_step'][step_idx]:
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            step_idx += 1
            if step_idx >= len(config['lr_step']):
                step_idx = 0  # prevent index out of range error
            print 'Learning rate changed to:', learning_rate.get_value()

    if config['lr_policy'] == 'auto':
        if (epoch > 5) and (val_record[-3] - val_record[-1] <
                            config['lr_adapt_threshold']):
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            print 'Learning rate changed to::', learning_rate.get_value()

    return step_idx


def get_val_error_loss(rand_arr, shared_x, shared_y,
                       val_lmdb_path, image_mean_path,
                       flag_para_load,batch_size, validate_model,
                       flag_top_5=False):

    validation_losses = []
    validation_errors = []
    if flag_top_5:
        validation_errors_top_5 = []

    ##### get training and validation data from lmdb file
    val_lmdb_iterator = read_lmdb(batch_size, val_lmdb_path,image_mean_path)
    val_data_size = val_lmdb_iterator.total_number
    n_val_batches = val_data_size / batch_size
    print ('n_val_batches ', n_val_batches)

    for val_index in range(n_val_batches):
        (val_img,val_label) = val_lmdb_iterator.next()
        #val_img = val_img.astype('float32')
        # #not essential
        shared_x.set_value(val_img)
        shared_y.set_value(val_label)

        if flag_top_5:
            loss, error, error_top_5 = validate_model()
        else:
            loss, error = validate_model()

        # print loss, error
        validation_losses.append(loss)
        validation_errors.append(error)

        if flag_top_5:
            validation_errors_top_5.append(error_top_5)

    this_validation_loss = np.mean(validation_losses)
    this_validation_error = np.mean(validation_errors)
    if flag_top_5:
        this_validation_error_top_5 = np.mean(validation_errors_top_5)
        return this_validation_error, this_validation_error_top_5, this_validation_loss
    else:
        return this_validation_error, this_validation_loss


def get_rand3d():
    tmp_rand = np.float32(np.random.rand(3))
    tmp_rand[2] = round(tmp_rand[2])
    return tmp_rand

def thread_load(config):
    train_lmdb_path = config['train_lmdb_path'] 
    image_mean_path = config['image_mean_path']
    batch_size = 256
    train_lmdb_iterator = read_lmdb(batch_size, train_lmdb_path,image_mean_path)
    n_train_batches = train_lmdb_iterator.total_batch
    count = 0
    while True:
        if len(data_list) > 2000:
            continue
        count = count + 1
        if( count >= n_train_batches):
            count = 0
        x, y = train_lmdb_iterator.next()
        data_list.append((x,y))


def train_model_wrap(train_model, shared_x, shared_y, minibatch_index, batch_size):
    #train_lmdb_path = '/data/user/xiaotian/imagenet5/lmdb/ilsvrc12_train_lmdb/'
    #image_mean_path = '/data/user/xiaotian/imagenet5/lmdb/ilsvrc_2012_mean.npy'
    #train_lmdb_path = '/home/2T/imagenet/lmdb/ilsvrc12_train_lmdb/'
    #image_mean_path = '/home/2T/imagenet/lmdb/ilsvrc_2012_mean.npy'
    #train_lmdb_iterator = read_lmdb(batch_size, train_lmdb_path,image_mean_path)
    #train_data_size = train_lmdb_iterator.total_number
    #n_train_batches = train_data_size / batch_size
#    while True:
#        if len(data_list) > 0:
#            s = time.time()
#            (img,batch_label) = data_list.pop(0)
#            t1 = time.time()
#            shared_x.set_value(img,borrow=True)
#            shared_y.set_value(batch_label,borrow=True)
#            t2 = time.time()
#            cost_ij = train_model()
#            t3 = time.time()
#            perf[0] +=(t3-t2)
#            perf[1] +=(t2-t1)
#            perf[2] +=(t1-s)
#            #print ("img ",img[0][0][0][3])
#            return cost_ij

#    train_lmdb_iterator.set_cursor(minibatch_index)
#    (train_img,train_label) = train_lmdb_iterator.next()
#    train_img = train_img.astype('float32')
#    shared_x.set_value(train_img,borrow=True)
#    shared_y.set_value(train_label,borrow=True)
    cost_ij = train_model()
    return cost_ij
