import json
import numpy as np
import pandas as pd

''' 1000개의 이미지에 존재하는 obj_id(중복 X) '''


def adjColumn(imgCount):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        object = []
        for i in range(imgCount):
            imageDescriptions = data[i]["objects"]
            for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
                object.append(imageDescriptions[j]['object_id'])
        scene_obj_id = sorted(list(set(object)))

        return scene_obj_id


def createAdj(imageId, adjColumn):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
        # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦

        # i는 image id
        imageDescriptions = data[imageId]["relationships"]
        object = []
        subject = []

        adjMatrix = np.zeros((len(adjColumn), len(adjColumn)))
        data_df = pd.DataFrame(adjMatrix)
        data_df.columns = adjColumn
        data_df = data_df.transpose()
        data_df.columns = adjColumn

        # ralationship에 따른
        for q in range(len(object)):
            row = adjColumn.index(object[q])
            column = adjColumn.index(subject[q])
            adjMatrix[column][row] += 1

        return data_df, adjMatrix

