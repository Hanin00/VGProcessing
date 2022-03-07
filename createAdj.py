import json
import pprint
import numpy as np
import pandas as pd

# for문 개수 통일해서 columns 통일하기 -> 적은 경우 padding 넣어야함. (이때 column 값은?)


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

        print(data_df)


