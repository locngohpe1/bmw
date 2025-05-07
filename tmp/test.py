import math
import numpy as np
import time

def get_list():
    l = []
    for i in range(10):
        time.sleep(0.1)
        l.append(i)
    return l

s = time.time()
for i in get_list():
    pass
print(time.time() - s)

s = time.time()
l = get_list()
for i in l:
    pass
print(time.time() - s)
