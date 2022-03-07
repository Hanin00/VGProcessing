import json
from openpyxl import Workbook

# region, scene 의 obj_id가 동일할 때 name값도 동일한 지 확인
"""
 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
     # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦
 i는 image id
"""
obj_id = 1546458 #computer
#obj_id = 1546459 #tower
with open('../data/scene_graphs.json') as file:  # open json file
    data = json.load(file)

    object = []
    subject = []

    for i in range(len(data)):
        imageDescriptions = data[i]["relationships"]

        for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            if obj_id == imageDescriptions[j]['object_id'] :
                print("scene : ",imageDescriptions[j]["names"])
                break

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
                if obj_id == objectsPerOneRegion[g]["object_id"]:
                    print("region : ",objectsPerOneRegion[g]["name"])
                    break

