#! /usr/bin/python2.7
"""
    Directories functions
"""
import os, shutil

def _mkdir_recursive(path):
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        _mkdir_recursive(sub_path)
    if not os.path.exists(path):
        os.mkdir(path)

def CheckDir(path, CLEAN):
    dir = os.path.dirname(path)
    top_dir = os.path.dirname(dir)
    if (CLEAN == 'yes'):
        if os.path.exists(dir) is True:
            shutil.rmtree(top_dir)

    # make dir on designate path
    sub_path = os.path.dirname(dir)
    if (os.path.exists(sub_path) is False):
        os.mkdir(sub_path)

    _mkdir_recursive(dir)
