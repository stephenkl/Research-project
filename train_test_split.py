import random
import numpy as np
from collections import defaultdict


path = '/home/yi/Desktop/AFL/dataset_equal/'
num_cls = 'fix_5'

with open(path+'all.txt', 'r') as f:
    lines = f.readlines()

dl = defaultdict(list)
for line in lines:
    line = line.rstrip()
    #print(line)
    _, _, cls = line.split()
    dl[cls].append(line)

train = defaultdict(list)
val = defaultdict(list)
test = defaultdict(list)

for key in dl.keys():
    temp = dl[key]
    #print(temp)
    random.shuffle(temp)
    #print(temp)
    index = int(np.round(0.7*len(temp)))
    train[key] = temp[:index]
    #print(len(train[key]))
    temp = temp[index:]
    index = int(np.round(0.5 * len(temp)))
    val[key] = temp[:index]
    test[key] = temp[index:]
    #print(len(val[key]))
    #print(len(test[key]))

train_file = open(path+'train_'+num_cls+'_cls.txt', 'w')
train_lines = []
for tv in train.values():
    train_lines += tv
print(train_lines)
for tl in train_lines:
    newline = tl +'\n'
    train_file.write(newline)
train_file.close()

val_file = open(path+'val_'+num_cls+'_cls.txt', 'w')
val_lines = []
for vv in val.values():
    val_lines += vv
print(val_lines)
for vl in val_lines:
    newline = vl +'\n'
    val_file.write(newline)
val_file.close()

test_file = open(path+'test_'+num_cls+'_cls.txt', 'w')
test_lines = []
for tv in test.values():
    test_lines += tv
print(test_lines)
for tl in test_lines:
    newline = tl +'\n'
    test_file.write(newline)
test_file.close()

