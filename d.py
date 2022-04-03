import json
import numpy as np
import sys

testFile = open('./data/freObj.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
adjColumn = label  # 빈출 100 단어
imageId = 1



with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    # 각 이미지 별로 obj, relationship 가져와서 freObj와 일치하는 obj들 간의 인접 행렬을 만듦
    # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦

    # imgId의 relationship에 따른 objId, subjId list
    # i는 image id
    imageDescriptions = data[imageId]["relationships"]
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
    # objectId = data[imgId][""]
    objects = data[1]["objects"]
    objIdName = []

    for i in range(len(objects)):
        objectsId = objects[i]["object_id"]
        objectsName = objects[i]["names"]
        objIdName.append((objectsId, ''.join(objectsName)))

    # ObjectName은 list 형태임. 여러 개의 이름을 갖는 경우가 있음. 그래서 list에 있는지로 파악해야함
    dictionary = dict(objIdName)



    adjMatrix = np.zeros((len(adjColumn), len(adjColumn)))

   # np.set_printoptions(threshold=784, linewidth=np.inf)

    for i in range(len(dictionary)):
        for j in range(len(object)):
            objName = ''
            subName = ''
           # obj 또는 sub 둘 중 하나만 freObj에 있는 경우
            if object[j] in dictionary:
                objName = dictionary[object[j]]
                # if(objName in adjColumn) :
                #     print("objName : ",objName)
                #     adjI = adjColumn.index(objName)
                #     print("adjI : ", adjI)
                #   #  adjMatrix[adjI][adjI] += 1

            if subject[j] in dictionary:
                subName = dictionary[subject[j]]
                print("subName : ",subName)
                # if(subName in adjColumn) :
                #     print("subName : ",subName)
                #     adjJ = adjColumn.index(subName)
                #     print("adjJ : ", adjJ)
                #     adjMatrix[adjJ][adjJ] += 1

# obj-subj 모두 freObj에 있는 경우
            if (objName != '') & (subName != ''):
                print("둘 다 있음")
                if (objName in adjColumn) & (subName in adjColumn):
                    adjI = adjColumn.index(objName)
                    adjJ = adjColumn.index(subName)
                    print('adjI : ',adjI)
                    print('adjJ : ',adjJ)
                    adjMatrix[adjI][adjJ] += 1

