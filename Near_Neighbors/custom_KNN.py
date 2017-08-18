#NOTE: - k-nearest-neighbors can run in parallel!!
# - doesnt run well in TB of data, otherwise runs well
# - can use a radius for classification
# - can work on linear and non linear data
# - 

import numpy as np
from math import sqrt
from collections import Counter
import warnings
import pandas as pd
import random

#given a dataset, and a 'new feature', it will return the class that that new feature belongs to.
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k: #ex: dataset has only 2 values, so max 2 groups
        warnings.warn('K is set to a value less than total voting groups')
    distances = []
    for group in data:
        for features in data[group]:
            euclid_distance = np.linalg.norm(np.array(features)-np.array(predict)) # calculates distances
            distances.append([euclid_distance,group]) #appends list of distances for corresponding group

    #i[1] is the group. want to rank the distances. only care about distances to k.
    votes = [i[1] for i in sorted(distances)[:k]]
    # most_common is an array of list. gives most common group, and how many there are
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    #if(Counter(votes).most_common(1)[0][0] != Counter(votes).most_common(1)[0][1]):
        #print(Counter(votes).most_common(1))
    #print(Counter(votes).most_common(1))

    #print(vote_result, confidence)
    return vote_result, confidence

df = pd.read_csv('breast_cancer.txt')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True) #drop worthless column. messes up accuracy
full_data = df.astype(float).values.tolist() #casts values to floats
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))] # first 20%
test_data = full_data[-int(test_size*len(full_data)):] # last 20%


for i in train_data:# iterates through the shuffled breast cancer data
    train_set[i[-1]].append(i[:-1]) # -1 is the last column, either a 2 or a 4

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote,confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1 # expected result is same as observed result
        else:
            print('Confidence ', confidence*100,'%' )
            print('Actual', group, 'Custom KNN', vote)
        total +=1

print('Accuracy',correct/total) # roughly 96% accuracy
