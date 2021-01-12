# -*- coding: utf-8 -*-

import os
import datetime
import glob
import re
import shutil
import pickle

import binarybrain as bb


def get_date_string():
    # データ保存パス用の日付文字列を生成
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def is_date_string(text: str):
    ''' Check if the string is a date
        データ保存パス用の日付文字列かどうか判定
    
        Args:
            text (str): 判定する文字列
        
        Returns:
            Boolean.
    '''
    
    if re.fullmatch('[12][0-9]{3}[01][0-9][0-3][0-9]_[0-2][0-9][0-5][0-9][0-5][0-9]', text):
        return True
    return False
    
def get_latest_path(path: str) -> str:
    ''' Get latest data path
        最新のデータ保存パスを取得
    
        Args:
            path (str): 検索するパス
        
        Returns:
            見つかったパス. 見つからなければ None
    '''
    
    if not os.path.exists(path):
        return None
    
    files = os.listdir(path)
    dirs  = [f for f in files if os.path.isdir(os.path.join(path, f))]
    
    targets = []
    for d in dirs:
        if is_date_string(d):
            targets.append(d)
    
    if not targets:
        return None
    
    targets.sort(reverse=True)
    return os.path.join(path, targets[0])


def remove_old(path: str, keeps: int=-1):
    ''' Get latest data path
        最新のデータ保存パスを取得
    
        Args:
            path (str): 検索するパス
            keeps (int): 削除せずに残す数
    '''
    
    if keeps < 0:
        return
    
    files = os.listdir(path)
    dirs  = [f for f in files if os.path.isdir(os.path.join(path, f))]
    
    targets = []
    for d in dirs:
        if is_date_string(d):
            no_delete_file = os.path.join(path, d, '__no_delete__')
            if not os.path.exists(no_delete_file):
                targets.append(d)
    
    targets.sort(reverse=True)
    del targets[:keeps]
    
    for t in targets:
        shutil.rmtree(os.path.join(path, t))


def _save_net_file(path: str, name: str, net, file_format=None):
    if file_format == 'bin' or file_format=='all':
        # 旧バージョン
        net_file_name = os.path.join(path, name + '.bin')
        with open(net_file_name, 'wb') as f:
            f.write(net.dump_bytes())
    
    elif file_format == 'pickle' or file_format=='all':
        # pickle
        net_file_name = os.path.join(path, name + '.pickle')
        with open(net_file_name, 'wb') as f:
            f.write(pickle.dumps(net))
    else:
        # デフォルトフォーマット
        net_file_name = os.path.join(path, name + '.bb_net')
        with open(net_file_name, 'wb') as f:
            f.write(net.dumps())

def _load_net_file(path: str, name: str, net, file_format=None) -> bool:
    # デフォルトフォーマット
    if file_format is None or file_format == 'bb_net':
        net_file_name = os.path.join(path, name + '.bb_net')
        if os.path.exists(net_file_name):
            with open(net_file_name, 'rb') as f:
                net.loads(f.read())
            return True
    
    # 無ければ旧フォーマットを探してみる
    if file_format is None or file_format == 'bin':
        net_file_name = os.path.join(path, name + '.bin')
        if os.path.exists(net_file_name):
            with open(net_file_name, 'rb') as f:
                net.load_bytes(f.read())
            return True

    # pickle 
    if file_format is None or file_format == 'pickle':
        net_file_name = os.path.join(path, name + '.pickle')
        if os.path.exists(net_file_name):
            with open(net_file_name, 'rb') as f:
                tmp_net = pickle.loads(f.read())
            # pickle はインスタンスが作り直されてしまうのでコピー
            net.loads(tmp_net.dumps())
            return True

    return False


def save_models(path: str, net, *, write_layers=True, file_format=None):
    ''' save networks
        ネットを構成するモデルの保存
        
        Args:
            path (str):  保存するパス
            net (Model): 保存するネット
            write_layers (bool) : レイヤー別にも出力するかどうか
    '''
    
    # make dir
    os.makedirs(path, exist_ok=True)
    
    # save
    net_name = net.get_name()
    _save_net_file(path, net_name, net, file_format=file_format)

    # save flatten models
    if write_layers:
        models = bb.get_model_list(net, flatten=True)
        fname_list = []  # 命名重複回避用
        for i, model in enumerate(models):
            name = model.get_name()
            if model.is_named():
                if name in fname_list:
                    print('[warrning] duplicate model name : %s', name)
                    fname = '%04d_%s' % (i, name)
                else:
                    fname = '%s' % (name)
            else:
                fname = '%04d_%s' % (i, name)
            fname_list.append(fname)
            
            _save_net_file(path, fname, model, file_format=file_format)
            


def load_models(path: str, net, *, read_layers: bool=False, file_format=None):
    ''' load networks
        ネットを構成するモデルの保存
        
        Args:
            path (str):  読み出すパス
            net (Model): 読み込むネット
            read_layers (bool) : レイヤー別に読み込むか
    '''

    # load
    if not read_layers:
        net_name = net.get_name()
        res = _load_net_file(path, net_name, net, file_format=file_format)
        if not res:
            print('file not found : %s'%os.path.join(path, net_name))
        return

    # load models
    models    = bb.get_model_list(net, flatten=True)
    fname_list = []
    for i, model in enumerate(models):
        name = model.get_name()
        if model.is_named():
            if name in fname_list:
                print('[warrning] duplicate model name : %s', name)
                fname = '%04d_%s' % (i, name)
            else:
                fname = '%s' % (name)
        else:
            fname = '%04d_%s' % (i, name)
        fname_list.append(fname)
        
        res = _load_net_file(path, fname, model, file_format=file_format)
        if not res:
            print('file not found : %s' % fname)


def save_networks(path: str, net, *, backups: int=3, write_layers: bool=False, file_format=None):
    ''' save networks
        ネットを構成するモデルの保存
        
        指定したパスの下にさらに日付でディレクトリを作成して保存
        古いものから削除する機能あり
        
        Args:
            path (str) : 保存するパス
            net (Model) : 保存するネット
            backups (int) : 残しておく古いデータ数
            write_layers(bool) : レイヤー別に出力
    '''
    
    # make dir
    os.makedirs(path, exist_ok=True)
    
    # save with date
    date_str = get_date_string()
    data_path = os.path.join(path, date_str)
    
    save_models(data_path, net, write_layers=write_layers, file_format=file_format)
    
    if backups >= 0:
        remove_old(path, keeps=backups)

def load_networks(path: str, net, *, read_layers: bool=False, file_format=None):
    ''' load network
        ネットを構成するモデルの読み込み
        
        最新のデータを探して読み込み
        
        Args:
            path (str) : 読み込むパス
            net (Model) : 読み込むネット
            file_format (str) : 読み込む形式(Noneがデフォルト)
    '''
    
    data_path = get_latest_path(path)
    if data_path is None:
        print('not loaded : file not found')
        return
    
    load_models(data_path, net, read_layers=read_layers, file_format=None)
    print('load : %s' % data_path)

