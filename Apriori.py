
# coding: utf-8

# # Table of Contents
#  <p>

# In[112]:

import random
import numpy as np
from itertools import combinations
from collections import defaultdict
from collections import Iterable


# In[115]:

class Apriori:
    def __init__(self,minSupport,numbers):
        self.numbers = numbers
        self.minSupport = minSupport
    def flattern(self,items,ignore_type=(tuple,)):
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x,ignore_type):
                yield from flattern(x)
            else:
                yield x
    def train(self,X):
        len_data = len(X)
        freq_parent = range(len_data)
        for i in range(1,self.numbers+1):
            data_use =  [(idx,combinations(raw_data[idx],i)) for idx in freq_parent]
            item_cnt = defaultdict(int)
            item_parent = defaultdict(list)
            for i,x in data_use:
                for y in x:
                    item_cnt[y] +=1
                    item_parent[y].append(i)
            freq_set = set([k for k,v in item_cnt.items() if v/len_data>self.minSupport])
            freq_parent = [item_parent.get(k) for k in freq_set ]
            freq_parent = set(self.flattern(freq_parent))
        return freq_set


# In[120]:

# 生成测试数据，测试
if __name__ == '__main__':
    data_set = []
    for i in range(100):
        data_set.append(random.sample(range(100),random.randint(1,5)))
    for a in range(3):
        data_setnew =  []
        for i in data_set:
            box = []
            for j in i:
                box.append(j)
                if (j%10)<5:
                    box.append(j+1)
            data_setnew.append(box)
        data_set = data_setnew
    raw_data = [set(i) for i in data_set]
    Ap = Apriori(0.05,4)
    freq_set = Ap.train(raw_data)

