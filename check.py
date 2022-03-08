import util as ut
import json
import numpy as np
import pandas as pd

#단어 빈도수 기반 임베딩을 하고 있음.. 되는거 하자 되는거.

img_id = 1
#img_cnt = 1000
#adjColumn, xWords = ut.adjColumn_kv(img_cnt)

with open('./data/xWords.txt', "r") as file:
    strings = file.readlines()[0]
    xWords = strings.split(',')

from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer()
tf = vector.fit_transform(xWords)
print(tf)
print(tf.toarray())
print(tf.shape) # (32121,2330) 확인함
#
# print(vector.fit_transform(xWords).toarray())
# print(vector.vocabulary_)
#
# print(vector.vocabulary_.get('shade'))
# xEmb = []
#
# for i in range(len(xWords)) :
#     print(xWords[i])
# print(xEmb)

# # i로 호출 후 list 에 저장 -> id 다른데 값 동일한 경우 있음.
# for i in range(xWords) :
#     if i == vector.vocabulary_.get(i) :
#         xEmb += vector.vocabulary_.get(i)
# print(xEmb)