import json
import numpy as np
import pandas as pd

"""
    object명 set으로 안겹치게 column 명... 걍 object id 중에서 cite 많이 된 순서로.
    -> 이미지 1000개의 list(set(object id)) -> column 명 
"""



''' 1000개의 이미지에 존재하는 obj_id(중복 X) '''
with open('../data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    object = []
    for i in range(1000):
        imageDescriptions = data[i]["objects"]
        for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            object.append(imageDescriptions[j]['object_id'])
    scene_obj = sorted(list(set(object)))

    print(scene_obj)

    print("scene_graph_obj_len : ", len(scene_obj))

# 인접행렬 반환
with open('../data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
    # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦
    # i는 image id
    i = 0
    imageDescriptions = data[i]["relationships"]
    object = []
    subject = []

    """
    for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
        object.append(imageDescriptions[j]['object_id'])
        subject.append(imageDescriptions[j]['subject_id'])
    adjColumn = list(set(object + subject))
    print(adjColumn[0])
    """

    adjMatrix = np.zeros((len(scene_obj), len(scene_obj)))
    data_df = pd.DataFrame(adjMatrix)
    data_df.columns = scene_obj
    data_df=data_df.transpose()
    data_df.columns = scene_obj

    # ralationship에 따른
    for q in range(len(object)):
        row = scene_obj.index(object[q])
        column = scene_obj.index(subject[q])
        adjMatrix[column][row] += 1

    print(data_df)