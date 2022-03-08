import util as ut
import json
import numpy as np
import pandas as pd

#단어 빈도수 기반 임베딩을 하고 있음.. 되는거 하자 되는거.

img_id = 1
img_cnt = 1000
adjColumn, xWords = ut.adjColumn_kv(img_cnt)


print("adjColumn : ", len(adjColumn))
print("xWords : ", len(xWords))

with open('./data/xWords.txt', "r") as file:
    strings = file.readlines()[0]
    xWords = strings.split(',')
print(len(xWords))

from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer()
tf = vector.fit_transform(xWords)
tfArray = tf.toarray()
print(tf)
print(tf.toarray())
print(tf.shape) # (32121,2330) 확인함
print(tfArray[0])
print(type(tfArray[0]))
print(tf[0])
print(type(tf[0]))

print('indices:', tf.indices)

xEmbedding = tf.indices
print(xEmbedding)
print(type(xEmbedding))


