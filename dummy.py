import csv
import json
import numpy as np
import util2 as ut




adjColumn = ut.adjColumn(1000)




'''
adjColumn = ["man", "window", "person", "tree", "building", "shirt", "wall", "woman",
             "sign", "sky", "ground", "light", "grass", "clud", "pole", "car", "table",
             "leaf", "hand", "leg", "head", "water", "hair", "people", "ear",
             "eye", "shoe", "plate", "flower", "line", "wheel", "door",
             "glass", "chair", "letter", "pant", "fence", "train", "floor",
             "street", "road", "hat", "shadow", "snow", "jacket",
             "boy", "boat", "rock", "handle"]

with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
    # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦

    # imgId의 relationship에 따른 objId, subjId list
    # i는 image id
    imageDescriptions = data[1]["relationships"]
    object = []
    subject = []

    for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
        object.append(imageDescriptions[j]['object_id'])
        subject.append(imageDescriptions[j]['subject_id'])


with open('./data/objects.json') as file:  # open json file
    data = json.load(file)
    # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
    # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦

    # imgId의 relationship에 따른 objId, subjId list
    # i는 image id
    #objectId = data[imgId][""]
    objects = data[1]["objects"]
    objIdName = []

    for i in range (len(objects)) :
        objectsId = objects[i]["object_id"]
        objectsName = objects[i]["names"]
        objIdName.append((objectsId, objectsName))

    #ObjectName은 list 형태임. 여러 개의 이름을 갖는 경우가 있음. 그래서 list에 있는지로 파악해야함
    dictionary = dict(objIdName)
    #print(dictionary[1023841])
    print(dictionary)

    adjMatrix = np.zeros((len(adjColumn), len(adjColumn)))

    for i in range(len(dictionary)):
        for j in range(len(object)):
            if object[j] in dictionary :
                objName = dictionary[object[j]]
            if subject[j] in dictionary :
                subName= dictionary[subject[j]]

        if (objName != '') & (subName!='') :
            if (objName in adjColumn) & (subName in adjColumn) :
                adjI = adjColumn.index(objName)
                adjJ = adjColumn.index(subName)
                adjMatrix[adjI][adjJ] += 1

        objName = ''
        subName = ''

    print(adjMatrix)
'''