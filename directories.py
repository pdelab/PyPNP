#! /usr/bin/python2.7
"""
    Directories functions
"""
import os


def CheckDir(DIR, CLEAN):
    if (CLEAN == 'yes'):
        if os.path.exists(DIR) is True:
            for root, dirs, files in os.walk(DIR, topdown=False):
                for clean_name in files:
                    os.remove(root+"/"+clean_name)
                for clean_name in dirs:
                    os.rmdir(root+"/"+clean_name)

    # make directory on designate path
    if (os.path.exists(DIR) is False):
        os.mkdir(DIR)
