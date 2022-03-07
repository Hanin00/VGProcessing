import json
import pprint
import numpy as np
import pandas as pd

# for문 개수 통일해서 columns 통일하기 -> 적은 경우 padding 넣어야함. (이때 column 값은?)

"""
    한 이미지당 모든 object 가져와서 리스트로 만들고 embedding(word2vec)
    인접행렬(relationship) 과 feature(object id) 행렬 만들기 -> 
    object명 set으로 안겹치게 column 명... 걍 object id 중에서 cite 많이 된 순서로.
    
    
    Column, Row 명이 object_id인 df 및 adjMatrix 반환
"""

def createAdj(imageId) :
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        pp = pprint.PrettyPrinter(indent=4)
        # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
        # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦

        # i는 image id
        imageDescriptions = data[imageId]["relationships"]
        object = []
        subject = []

        for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            object.append(imageDescriptions[j]['object_id'])
            subject.append(imageDescriptions[j]['subject_id'])

        adjColumn = list(set(object + subject))

        adjMatrix = np.zeros((len(adjColumn), len(adjColumn)))
        data_df = pd.DataFrame(adjMatrix)

        # ralationship에 따른
        for q in range(len(object)):
            row = adjColumn.index(object[q])
            column = adjColumn.index(subject[q])
            adjMatrix[column][row] += 1

        return data_df, adjMatrix

