# Read-LMDB-Without-CAFFE(ALEXNET)
use ilsvrc12 lmdb datasets without installing caffe in theano.

Model : AlexNet.
* use lmdb datasets encoded by caffe
* without installing caffe

# Use Intel Theano & MKL
https://github.com/intel/Theano

# Time on Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz

> Average Forward-Backward pass: 610.16 ms

> Average Forward pass: 218.17 ms

# How to use independently in other Model
* copy caffe_pb2.py & new_lmdb.py to your directory

* use interface template:

```  
from new_lmdb import *    
train_lmdb_iterator = read_lmdb(batch_size, train_lmdb_path,image_mean_path)

(train_img,train_label) = train_lmdb_iterator.next()
shared_x.set_value(train_img,borrow=True)
shared_y.set_value(train_label,borrow=True)
minibatch_index = 0
cost_ij = train_model_wrap(train_model, shared_x,
                             shared_y,minibatch_index,
                             batch_size)
```  
* The file type of image_mean_path must be *.npy* .
