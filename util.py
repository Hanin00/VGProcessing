import numpy as np
import pandas as pd
import json
from openpyxl import Workbook
import util as ut


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

''' adj 생성(이미지 하나에 대한) '''
def createAdj(imageId, adjColumn):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
        # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦

        # i는 image id
        imageDescriptions = data[imageId]["relationships"]
        object = []
        subject = []

        for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            object.append(imageDescriptions[j]['object_id'])
            subject.append(imageDescriptions[j]['subject_id'])

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


''' 
Y data 생성을 위해 image에 대한 text description을 이미지 별로 모음

jsonpath : './data/region_descriptions.json'
xlxspath : './data/image_regions.xlsx'
'''
def jsontoxml(imgCnt, jsonpath, xlsxpath) :
    with open(jsonpath) as file:  # open json file
        data = json.load(file)
        wb = Workbook()  # create xlsx file
        ws = wb.active  # create xlsx sheet
        ws.append(['image_id', 'region_sentences'])

        phrase = []

        q = 0
        for i in data:
            if q == imgCnt:
                break
            regions = i.get('regions')
            imgId = regions[0].get('image_id')
            k = 0
            for j in regions:
                if k == 7:
                    break
                phrase.append(j.get('phrase'))
                k += 1
            sentences = ','.join(phrase)
            ws.append([imgId, sentences])
            phrase = []
            q += 1
        wb.save(xlsxpath)


''' 1000개의 이미지에 존재하는 obj_name(중복 X) > Featuremap object_name Embedding 원본'''
def adjColumn_kv(imgCount):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        dict = {}

        # dict 생성(obj_id : obj_name)
        for i in range(imgCount):
            imageDescriptions = data[i]["objects"]
            for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
                obj_id = imageDescriptions[j]['object_id']
                obj_name = imageDescriptions[j]['names']
                dict[obj_id] = obj_name

        keys = sorted(dict)
        val = []

        for i in keys:
            val += dict[i]

        return keys, val