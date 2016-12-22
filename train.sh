#!/bin/bash
source ~/.bashrc
rm -r /root/.theano/compi*/* -rf
python train.py

