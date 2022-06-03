from collections import Counter
import matplotlib.pyplot as plt
from turtle import color
import pandas as pd 

def get_color(cluster):
    color_arr = ['red', 'green', 'darkorange', 'blue']
    return color_arr[cluster]

def get_intervals (K, start, ci_buf, label_results):
    buf = [None]*K
    for pos, cluster in enumerate(label_results):
        ci = ci_buf[pos + start]
        if buf[cluster] == None:
            buf[cluster] = (ci, ci)
        else:
            _min, _max = buf[cluster]
            if (ci < _min):
                _min = ci
            if (ci > _max):
                _max = ci
            buf[cluster] = (_min, _max)
    buf = sorted(buf, key=lambda tup:tup[0])
    return buf
