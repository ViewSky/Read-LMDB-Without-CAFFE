import sys
import time
from multiprocessing import Process, Queue
import thread

import yaml
import numpy as np
#import zmq
from new_lmdb import *
# Jinlong added: 2015-10-27
import logging
# set up logging to file - see previous section for more details

from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    filename='./alexnet_time_tmp.log',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

# Now, define a couple of other loggers which might represent areas in your
# application:
logger = logging.getLogger('AlexNet.timming')

sys.path.append('./lib')
from tools import (save_weights, load_weights,
                   save_momentums, load_momentums)
from train_funcs import (unpack_configs, adjust_learning_rate,
                         get_val_error_loss, get_rand3d, thread_load,
                         train_model_wrap,proc_configs)


def train_net(config):
    # UNPACK CONFIGS
    (flag_para_load, train_lmdb_path, val_lmdb_path, image_mean_path) = unpack_configs(config)
    import theano
    theano.config.on_unused_input = 'warn'

    if config['flag_top_5']:    # Jinlong added 2015-10-30
        flag_top5 = True
    else:
        flag_top5 = False

    from layers import DropoutLayer
    from alex_net import AlexNet, compile_models

    ## BUILD NETWORK ##
    model = AlexNet(config)
    layers = model.layers
    batch_size = model.batch_size

    ## COMPILE FUNCTIONS ##
    (train_model, validate_model, train_error, learning_rate,
        shared_x, shared_y, rand_arr, vels) = compile_models(model, config, flag_top_5=flag_top5)


    ######################### TRAIN MODEL ################################

    print '... training'


#############################################
    #train_lmdb_path = '/data/user/xiaotian/imagenet5/lmdb/ilsvrc12_train_lmdb/'
    #image_mean_path = '/data/user/xiaotian/imagenet5/lmdb/ilsvrc_2012_mean.npy'
    #train_lmdb_path = '/home/2T/imagenet/lmdb/ilsvrc12_train_lmdb/'
    #image_mean_path = '/home/2T/imagenet/lmdb/ilsvrc_2012_mean.npy'
    train_lmdb_iterator = read_lmdb(batch_size, train_lmdb_path,image_mean_path)
    train_data_size = train_lmdb_iterator.total_number
    n_train_batches = train_data_size / batch_size
    minibatch_range = range(n_train_batches)
    train_lmdb_iterator.lmdb_cursor.first()
    print "train batches:", n_train_batches

############################################

    # Start Training Loop
    epoch = 0
    step_idx = 0
    val_record = []

    while epoch < config['n_epochs']:
        epoch = epoch + 1

        if config['shuffle']:
	    print ('shuffle')
            np.random.shuffle(minibatch_range)

        if config['resume_train'] and epoch == 1:
            train_lmdb_iterator.lmdb_cursor.first()
            (train_img,train_label) = train_lmdb_iterator.next()
            train_lmdb_iterator.lmdb_cursor.first()
            #print "train_label:",train_label
            shared_x.set_value(train_img,borrow=True)
            shared_y.set_value(train_label,borrow=True)
            minibatch_index = 0
            cost_ij = train_model_wrap(train_model, shared_x,
                                       shared_y,minibatch_index,
                                       batch_size)
	    print ('config')
            load_epoch = config['load_epoch']
            load_weights(layers, config['weights_dir'], load_epoch)
            #print layers[0].params[1].get_value()
            #sys.exit(0)
            epoch = load_epoch + 1
            lr_to_load = np.load(
                config['weights_dir'] + 'lr_' + str(load_epoch) + '.npy')
            #val_record = list(
            #    np.load(config['weights_dir'] + 'val_record.npy'))
            learning_rate.set_value(lr_to_load)
            load_momentums(vels, config['weights_dir'], load_epoch)
            #load_momentums(vels, config['weights_dir'], epoch)

            DropoutLayer.SetDropoutOff()
    
            # result_list = [ this_validation_error, this_validation_loss ]
            result_list = get_val_error_loss(
            #this_validation_error, this_validation_loss = get_val_error_loss(
                rand_arr, shared_x, shared_y,
                val_lmdb_path, image_mean_path,
                flag_para_load, 
                batch_size, validate_model,
                flag_top_5=flag_top5)
    
    
            logger.info(('epoch %i: validation loss %f ' %
                  (epoch, result_list[-1])))
            #print('epoch %i: validation loss %f ' %
            #      (epoch, this_validation_loss))
            if flag_top5:
                logger.info(('epoch %i: validation error (top 1) %f %%, (top5) %f %%' %
                    (epoch,  result_list[0] * 100., result_list[1] * 100.)))
            else:
                logger.info(('epoch %i: validation error %f %%' %
                    (epoch, result_list[0] * 100.)))
            #print('epoch %i: validation error %f %%' %
            #      (epoch, this_validation_error * 100.))
            val_record.append(result_list)
            #val_record.append([this_validation_error, this_validation_loss])
            np.save(config['weights_dir'] + 'val_record.npy', val_record)
    
            DropoutLayer.SetDropoutOn()



        count = 0
        #import proc_load
        #for minibatch_index in minibatch_range[:1]:
        for minibatch_index in minibatch_range:
            num_iter = (epoch - 1) * n_train_batches + count
            count = count + 1
            #print str(datetime.today()) + "  @iter " + str(count)
            if count == 1:
                s = time.time()
            if count % 20 == 0:
                e = time.time()
                print "time per 20 iter:", (e - s)
                logger.info("time per 20 iter: %lf" % (e - s))
                s = e
            ########################
            (train_img,train_label) = train_lmdb_iterator.next()
            #print "train_label:",train_label
            shared_x.set_value(train_img)
            shared_y.set_value(train_label)
            #print "train_label:",shared_y.get_value()
#########################
            cost_ij = train_model_wrap(train_model, shared_x,
                                       shared_y,minibatch_index,
                                       batch_size)
    #        print ("compute:%f, set_value:%f, pop: %f ms\n"%(1000.0*perf[0]/20,1000.0*perf[1]/20,1000.0*perf[2]/20))
            if num_iter % config['print_freq'] == 0:
		logger.info("training @ iter = %i" % (num_iter))
		logger.info("training cost: %lf, lr:%lf" % (cost_ij,learning_rate.get_value()))
                if config['print_train_error']:
                    logger.info('training error rate: %lf' % train_error())
                    #print 'training error rate:', train_error()

        ############### Test on Validation Set ##################
        continue
        #"""
        DropoutLayer.SetDropoutOff()

        # result_list = [ this_validation_error, this_validation_loss ]

        result_list = get_val_error_loss(
            #this_validation_error, this_validation_loss = get_val_error_loss(
                rand_arr, shared_x, shared_y,
                val_lmdb_path, image_mean_path,
                flag_para_load, 
                batch_size, validate_model,
                flag_top_5=flag_top5)

        logger.info(('epoch %i: validation loss %f ' %
              (epoch, result_list[-1])))
        #print('epoch %i: validation loss %f ' %
        #      (epoch, this_validation_loss))
        if flag_top5:
            logger.info(('epoch %i: validation error (top 1) %f %%, (top5) %f %%' %
                (epoch,  result_list[0] * 100., result_list[1] * 100.)))
        else:
            logger.info(('epoch %i: validation error %f %%' %
                (epoch, result_list[0] * 100.)))
        #print('epoch %i: validation error %f %%' %
        #      (epoch, this_validation_error * 100.))
        val_record.append(result_list)
        #val_record.append([this_validation_error, this_validation_loss])
        np.save(config['weights_dir'] + 'val_record.npy', val_record)

        DropoutLayer.SetDropoutOn()
        ############################################

        # Adapt Learning Rate
        step_idx = adjust_learning_rate(config, epoch, step_idx,
                                        val_record, learning_rate)

        # Save weights
        if epoch % config['snapshot_freq'] == 0:
            save_weights(layers, config['weights_dir'], epoch)
            np.save(config['weights_dir'] + 'lr_' + str(epoch) + '.npy',
                       learning_rate.get_value())
            save_momentums(vels, config['weights_dir'], epoch)
        #"""

    print('Optimization complete.')


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    #with open('spec.yaml', 'r') as f:
    #    config = dict(config.items() + yaml.load(f).items())

    config = proc_configs(config)

    if config['para_load']:
        #from proc_load import fun_load
        #config['queue_l2t'] = Queue(1)
        #config['queue_t2l'] = Queue(1)
        #print ("train\n")
        ##train_proc = Process(target=train_net, args=(config,))
	#thread.start_new_thread(train_net,(config,))
        #thread.start_new_thread(fun_load,(config,config['sock_data']))
        thread.start_new_thread(thread_load,(config,))
        
        train_net(config)
        #load_proc = Process(target=fun_load, args=(config, config['sock_data']))
        #train_proc.start()
        #load_proc.start()
        #train_proc.join()
        #load_proc.join()

    else:
        """
        train_proc = Process(target=train_net, args=(config,))
        train_proc.start()
        train_proc.join()
        """
        train_net(config) # For theano profiling
