﻿# coding: utf-8

import os
import sys
from collections import OrderedDict
import binarybrain as bb
import numpy as np


#if 'ipykernel' in sys.modules:
#   from tqdm import tqdm_notebook as tqdm
#else:
#   from tqdm import tqdm

from tqdm import tqdm


def calculation(net, x, x_shape, t, t_shape, max_batch_size, min_batch_size=1,
            metrics=None, loss=None, optimizer=None, train=False,
            print_loss=True, print_metrics=True, leave=False):
    
    if metrics is not None:
        metrics.clear()
    
    if loss is not None:
        loss.clear()
    
    batch_size = len(x)
    
    x_buf = bb.FrameBuffer()
    t_buf = bb.FrameBuffer()
    
#   for index in tqdm(range(0, batch_size, max_batch_size)):
    with tqdm(range(0, batch_size, max_batch_size), leave=leave) as pbar:
        for index in pbar:
            # calc mini_batch_size
            mini_batch_size = min(max_batch_size, batch_size-index)
            
            # setup x
            x_buf.resize(mini_batch_size, x_shape)
            x_buf.set_data(x[index:index+mini_batch_size])
            
            # forward
            y_buf = net.forward(x_buf, train)
            
            # setup t
            t_buf.resize(mini_batch_size, t_shape)
            t_buf.set_data(t[index:index+mini_batch_size])
            
            # calc loss
            if loss is not None:
                dy_buf = loss.calculate_loss(y_buf, t_buf, mini_batch_size)

            # calc metrics
            if metrics is not None:
                metrics.calculate_metrics(y_buf, t_buf)

            # backward
            if train and loss is not None:
                net.backward(dy_buf)

                # update
                if  optimizer is not None:
                    optimizer.update()
            
            # print progress
            dict = OrderedDict()
            if print_loss and loss is not None:
                dict['loss'] = loss.get_loss()
            if print_metrics and metrics is not None:
                dict[metrics.get_metrics_string()] = metrics.get_metrics()
            if len(dict) > 0:
                pbar.set_postfix(dict)

class Runner:
    def __init__(
            self,
            net,
            name="",
            loss=None,
            metrics=None,
            optimizer=None,
            max_run_size=0,
            print_progress=True,
            print_progress_loss=True,
            print_progress_accuracy=True,
            log_write=True,
            log_append=True,
            data_augmentation=None
            ):
        self.net                     = net
        self.name                    = name
        self.loss                    = loss
        self.metrics                 = metrics
        self.optimizer               = optimizer
        self.max_run_size            = max_run_size
        self.print_progress          = print_progress
        self.print_progress_loss     = print_progress_loss
        self.print_progress_accuracy = print_progress_accuracy
        self.log_write               = log_write
        self.log_append              = log_append
        self.data_augmentation       = data_augmentation
    
    def fitting(self, td, epoch_size, mini_batch_size=16, file_read=False, file_write=False, write_serial=False, init_eval=False):
        """fitting
        
        Args:
            td (TrainData)  : training data set
            epoch_size (int): epoch size
            mini_batch_size (int): mini batch size
        """
      
        log_file_name  = self.name + '_log.txt'
        json_file_name = self.name + '_net.json'
        epoch = 0
        
        # read
        if file_read:
            ret = bb.RunStatus.ReadJson(json_file_name, self.net, self.name, epoch)
            if ret:
                print('[load] %s'% json_file_name)
            else:
                print('[file not found] %s'% json_file_name)
        
        x_train  = np.array(td['x_train'])
        t_train  = np.array(td['t_train'])
        x_test   = np.array(td['x_test'])
        t_test   = np.array(td['t_test'])
        x_shape  = td['x_shape']
        t_shape  = td['t_shape']
        
        # write network info
        with open(log_file_name, 'a') as log_file:
            print(self.net.get_info(), file=log_file)
        
        # initial evaluation
        if init_eval:
            calculation(self.net, x_test, x_shape, t_test, t_shape, mini_batch_size, 1, self.metrics, self.loss)
            output_text = '[initial] test_%s=%f test_loss=%f' % (self.metrics.get_metrics_string(), self.metrics.get_metrics(), self.loss.get_loss())
            
            calculation(self.net, x_train, x_shape, t_train, t_shape, mini_batch_size, 1, self.metrics, self.loss)
            output_text += ' train_%s=%f train_loss=%f' % (self.metrics.get_metrics_string(), self.metrics.get_metrics(), self.loss.get_loss())
            
            print(output_text)
            with open(log_file_name, 'a') as log_file:
                print(output_text, file=log_file)
        
        # loop
        result_list = []
        for _ in range(epoch_size):
            # increment
            epoch = epoch + 1
            
            if self.data_augmentation is not None:
                x_train_tmp, t_train_tmp = self.data_augmentation(x_train.copy(), t_train.copy(), x_shape, t_shape)
            else:
                x_train_tmp, t_train_tmp = x_train, t_train
            
            # train
            calculation(self.net, x_train_tmp, x_shape, t_train_tmp, t_shape, mini_batch_size, mini_batch_size,
                        self.metrics, self.loss, self.optimizer, train=True, print_loss=True, print_metrics=True)
            
            # write file
            if file_write:
                ret = bb.RunStatus.WriteJson(json_file_name, self.net, self.name, epoch)
                if not ret:
                    print('[write error] %s'% json_file_name)
            
            
            # evaluation
            output_text  = 'epoch=%d' % epoch
            
            calculation(self.net, x_test, x_shape, t_test, t_shape, mini_batch_size, 1, self.metrics, self.loss)
            test_metrics = self.metrics.get_metrics()
            test_loss    = self.loss.get_loss()
            output_text += ' test_%s=%f test_loss=%f' % (self.metrics.get_metrics_string(), test_metrics, test_loss)
            
            calculation(self.net, x_train, x_shape, t_train, t_shape, mini_batch_size, 1, self.metrics, self.loss)
            train_metrics = self.metrics.get_metrics()
            train_loss    = self.loss.get_loss()
            output_text += ' train_%s=%f train_loss=%f' % (self.metrics.get_metrics_string(), train_metrics, train_loss)
            
            result_list.append([test_metrics, test_loss, train_metrics, train_loss])
            
            print(output_text)
            with open(log_file_name, 'a') as log_file:
                print(output_text, file=log_file)
            
            # shuffle
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            t_train = t_train[p]
        
        return result_list
    
    
    def evaluation(self, td, mini_batch_size=16):
        """evaluation
        
        Args:
            td (TrainData): data set
            mini_batch_size (int): mini batch size
        """
        
        calculation(self.net, td['x_test'], td['x_shape'], td['t_test'], td['t_shape'], mini_batch_size, 1, self.metrics, self.loss)
        print('%s=%f loss=%f' % (self.metrics.get_metrics_string(), self.metrics.get_metrics(), self.loss.get_loss()))



import cv2

def image_data_augmentation(
        shift_x_range=0.0,
        shift_y_range=0.0,
        flip_x_rate=0.0,
        flip_y_rate=0.0,
        rotation_range=0.0,
        scale_range=0.0,
        neg_rate=0.0,
        binarize=False,
        rate = 1.0,
        ):
    
    def data_augmentation(x_vec, t_vec, x_shape, t_shape):
        width  = x_shape[0]
        height = x_shape[1]
        np_shape = list(reversed(x_shape))
        for i, x in enumerate(x_vec):
            if np.random.rand() < rate:
                x = x.reshape(np_shape)
                shift_x = (np.random.rand() * 2.0 - 1.0) * shift_x_range * width
                shift_y = (np.random.rand() * 2.0 - 1.0) * shift_y_range * height
                flip_x  = np.random.rand() < flip_x_rate
                flip_y  = np.random.rand() < flip_y_rate
                neg     = np.random.rand() < neg_rate
                angle   = (np.random.rand() * 2.0 - 1.0) * rotation_range
                scale   = (np.random.rand() * 2.0 - 1.0) * scale_range + 1.0
                mat = cv2.getRotationMatrix2D((x_shape[0]/2, x_shape[1]/2), angle , scale)
                th  = np.random.rand()
                mat[0][2] += shift_x
                mat[1][2] += shift_y
                for j, p in enumerate(x):
                    if flip_x:  p = cv2.flip(p, 1)
                    if flip_y:  p = cv2.flip(p, 0)
                    if neg:     p = 1.0 - p
                    p = cv2.warpAffine(p, mat, (x_shape[0], x_shape[1]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
                    if binarize:
                        _, p = cv2.threshold(p, th, 1.0, cv2.THRESH_BINARY)
                    x[j] = p
                x_vec[i] = x.reshape(-1)
 
        return x_vec, t_vec
    
    return data_augmentation
