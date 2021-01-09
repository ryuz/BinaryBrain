# -*- coding: utf-8 -*-

import binarybrain      as bb
import binarybrain.core as core


def get_version_string():
    """バージョン文字列取得
    
    Returns:
        version (str) : バージョン文字列
    """
    return core.get_version_string()


def omp_set_num_threads(threads: int):
    """omp_set_num_threads
    
       omp_set_num_threadsを呼び出す
       バックグランドで学習する場合など、Host側のCPUをすべて使うと
       逆に性能が落ちる場合や、運用上不便なケースなどで個数制限できる

    Args:
        threads (int) : OpenMPでのスレッド数
    """
    core.omp_set_num_threads(threads)


def is_device_available():
    """デバイス(GPU)が有効かの確認

    Returns:
        device_available (bool) : デバイス(GPU)が利用可能なら True を返す
    """
    return core.Manager.is_device_available()


def set_host_only(host_only: bool):
    """ホスト(CPU)のみの指定

        True を設定するとデバイス(GPU)を未使用としてホスト(CPU)のみを利用

    Args:
        host_only (bool) : ホストのみの場合 True を指定
    """
    core.Manager.set_host_only(host_only)

def get_device_count():
    """利用可能なデバイス(GPU)の個数を確認

    Returns:
        device_count (int) : 利用可能なデバイス(GPU)の個数を返す
    """
    return core.get_device_count()

def set_device(device_id):
    """利用するデバイス(GPU)を切り替え

    Args:
        device_id (int) : 利用するデバイス番号を指定
    """
    core.set_device(device_id)

def get_device_properties_string(device_id):
    """現在のデバイス(GPU)の情報を入れた文字列を取得

    Args:
        device_id (int) : 情報を取得するデバイス番号を指定

    Returns:
        device_properties_string (str) : 現在のデバイス(GPU)の情報を入れた文字列を返す
    """
    return core.get_device_properties_string(device_id)


def get_device_properties(device_id):
    return core.get_device_properties(device_id)

def get_device_name(device_id):
    return core.get_device_name(device_id)


