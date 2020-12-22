# -*- coding: utf-8 -*-

import os
import datetime
import glob
import re
import shutil

import binarybrain as bb


def get_data_string():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def get_latest_path(path):
    files = os.listdir(path)
    dirs  = [f for f in files if os.path.isdir(os.path.join(path, f))]
    
    targets = []
    for d in dirs:
        if re.match('[0-9]{8}_[0-9]{6}', d):
            targets.append(d)
    
    if not targets:
        return None
    
    targets.sort(reverse=True)
    return os.path.join(path, targets[0])


def remove_old(path, keep=-1):
    if keep < 0:
        return
    
    files = os.listdir(path)
    dirs  = [f for f in files if os.path.isdir(os.path.join(path, f))]
    
    targets = []
    for d in dirs:
        if re.match('[0-9]{8}_[0-9]{6}', d):
            targets.append(d)
    
    targets.sort(reverse=True)
    del targets[:keep]
    
    for t in targets:
        shutil.rmtree(os.path.join(path, t))

    
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
        
        print(file_path)
        with open(file_path, 'wb') as f:
            f.write(model.dump_bytes())

def save(path: str, net, keep_olds=-1):
    models = get_model_list(net, flatten=True)
    
    os.makedirs(path, exist_ok=True)
    
    date_str = get_data_string()
    data_path = os.path.join(path, date_str)
    os.makedirs(data_path, exist_ok=True)
    
    save_networls(data_path, net)
    
    if keep_olds >= 0:
        remove_old(path, keep=keep_olds)