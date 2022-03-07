import json
import numpy as np
import pandas as pd

with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
    # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦
    # i는 image id
    i = 0
    imageDescriptions = data[i]["relationships"]
    object = []
    subject = []

    for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
        object.append(imageDescriptions[j]['object_id'])
        subject.append(imageDescriptions[j]['subject_id'])
    adjColumn = list(set(object + subject))
    print(adjColumn[0])

    adjMatrix = np.zeros((len(adjColumn), len(adjColumn)))
    data_df = pd.DataFrame(adjMatrix)
    data_df.columns = adjColumn
    data_df=data_df.transpose()
    data_df.columns = adjColumn

    # ralationship에 따른
    for q in range(len(object)):
        row = adjColumn.index(object[q])
        column = adjColumn.index(subject[q])
        adjMatrix[column][row] += 1

    print(data_df)