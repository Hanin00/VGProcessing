import json
from openpyxl import Workbook

# region, scene graph에서 object_id set으로 만들고 count
"""
 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
     # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦
 i는 image id
"""

with open('../data/scene_graphs.json') as file:  # open json file
    data = json.load(file)

    object = []
    subject = []

    for i in range(len(data)):
        imageDescriptions = data[i]["relationships"]
        for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            object.append(imageDescriptions[j]['object_id'])
            subject.append(imageDescriptions[j]['subject_id'])
    scene_obj = list(set(object + subject))
    print("scene_graph_obj_len : ", len(scene_obj))

with open('../data/region_graphs.json') as file:  # open json file
    data = json.load(file)
    # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
    # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦
    # i는 image id
    object = []
    subject = []

    for k in range(len(data)):
        regionsPerOneImg = data[k]["regions"]  # 이미지의 object 개수만큼 반복
        for j in range(len(regionsPerOneImg)):
            objectsPerOneRegion = regionsPerOneImg[j]["objects"]
            for g in range(len(objectsPerOneRegion)):
                object.append(objectsPerOneRegion[g]["object_id"])
    # object.append(imageDescriptions[j]['objects'])

object_Obj = list(set(object))
print("object_Obj_len : ", len(object_Obj))
