# coding: utf-8

import os
import sys
from collections import OrderedDict
import binarybrain as bb


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
            seed=1):
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
        
        # log start
        with open(log_file_name, 'a') as log_file:
            # initial evaluation
            if init_eval:
                calculation(self.net, td['x_test'], td['x_shape'], td['t_test'], td['t_shape'], mini_batch_size, 1, self.metrics, self.loss)
                print('[initial] %s=%f loss=%f' % (self.metrics.get_metrics_string(), self.metrics.get_metrics(), self.loss.get_loss()))
            
            # loop
            for _ in range(epoch_size):
                # increment
                epoch = epoch + 1
                
                # train
                calculation(self.net, td['x_test'], td['x_shape'], td['t_test'], td['t_shape'], mini_batch_size, mini_batch_size,
                            self.metrics, self.loss, self.optimizer, train=True, print_loss=True, print_metrics=True)
                
                # write file
                if file_write:
                    ret = bb.RunStatus.WriteJson(json_file_name, self.net, self.name, epoch)
                    if not ret:
                        print('[write error] %s'% json_file_name)
                
                # evaluation
                calculation(self.net, td['x_test'], td['x_shape'], td['t_test'], td['t_shape'], mini_batch_size, 1, self.metrics, self.loss)
                output_text = 'epoch=%d %s=%f loss=%f' % (epoch, self.metrics.get_metrics_string(), self.metrics.get_metrics(), self.loss.get_loss())
                print(output_text)
                print(output_text, file=log_file)
    
    def evaluation(self, td, mini_batch_size=16):
        """evaluation
        
        Args:
            td (TrainData): data set
            mini_batch_size (int): mini batch size
        """
        
        calculation(self.net, td['x_test'], td['x_shape'], td['t_test'], td['t_shape'], mini_batch_size, 1, self.metrics, self.loss)
        print('%s=%f loss=%f' % (self.metrics.get_metrics_string(), self.metrics.get_metrics(), self.loss.get_loss()))


