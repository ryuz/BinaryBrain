# -*- coding: utf-8 -*-

import os
import datetime
import glob

import binarybrain as bb


def get_model_list(net, flatten=False):
    if  type(net) is not list:
        net = [net]
    
    if not flatten:
        return net
    
    def flatten_list(in_list, out_list):
        for model in in_list:
            if hasattr(model, 'get_model_list'):
                flatten_list(model.get_model_list(), out_list)
            else:
                out_list.append(model)
    
    out_list = []
    flatten_list(net, out_list)
    
    return out_list

def save_networls(path, net):
    models    = get_model_list(net, flatten=True)
    fname_list = []  # 命名重複回避用
    for i, model in enumerate(models):
        name = model.get_name()
        if model.is_named():
            if name in fname_list:
                print('[warrning] duplicate model name : %s', name)
                fname = '%04d_%s.bin' % (i, name)
            else:
                fname = '%s.bin' % (name)
        else:
            fname = '%04d_%s.bin' % (i, name)
        fname_list.append(fname)
        
        file_path = os.path.join(path, fname)
        
        with open(file_path, 'wb') as f:
            f.write(model.dump_bytes())

def save(path: str, net):
    models = get_model_list(net, flatten=True)
    
    os.makedirs(path, exist_ok=True)
    
    files = os.listdir(path)
    dirs  = [f for f in files if os.path.isdir(os.path.join(path, f))]
    
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    data_path = os.path.join(path, date_str)
    os.makedirs(data_path, exist_ok=True)
    
    save_networls(path, net)

