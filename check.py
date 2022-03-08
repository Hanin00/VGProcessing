import util as ut
import json
import numpy as np
import pandas as pd

img_id = 1
#img_cnt = 1000
#adjColumn, xWords = ut.adjColumn_kv(img_cnt)

with open('./data/adjColumn.txt', "r") as file:
    strings = file.readlines()[0]
    adjIdx = strings.split(',')

with open('./data/scene_graphs.json')as file:  # open json file
    data = json.load(file)
    # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
    # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦

    # i는 image id
    imageDescriptions = data[img_id]["relationships"]
    object = []
    subject = []

    for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
        object.append(imageDescriptions[j]['object_id'])
        subject.append(imageDescriptions[j]['subject_id'])

    adjMatrix = np.zeros((len(adjIdx), len(adjIdx)))

    # ralationship에 따른
    for q in range(len(object)):
        row = adjIdx.index(str(object[q]))
        column = adjIdx.index(str(subject[q]))
        #print("row : ", row)
        #print("column : ", column)
        adjMatrix[row][column] += 1.0

    print(adjMatrix[23][8]) #인접행렬에서 값 찍히는 것 확인함