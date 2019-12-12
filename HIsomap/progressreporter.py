from __future__ import division
import numpy as np


class call_callback:
    def __init__(self, callback):
        self.callback = callback
        self.oldp = -1

    def __call__(self, p):
        if p>self.oldp:
            self.oldp = p
            self.callback(p)

def noop(*args):
    pass

def progressreporter(callback=None):
    if callback:
        return call_callback(callback)
    else:
        return noop
