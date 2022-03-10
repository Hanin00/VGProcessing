import numpy as np
import pandas as pd
import json
from openpyxl import Workbook
import util as ut
from gensim.models import FastText
from tqdm import tqdm
import util as ut


np.set_printoptions(linewidth=np.inf)

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




''' obj name 단순 임베딩(fasttext로 임베딩 한 값)'''
def objNameEmbedding(xWords) :
    a = []
    a.append(xWords)

    model = FastText(a, vector_size=10, workers=4, sg=1, word_ngrams=1)

    # for i in a :
    embedding = []
    for i in xWords:
        embedding.append(model.wv[i])

    return embedding




''' 1000개의 이미지에 존재하는 obj_name(중복 X) > Featuremap object_name Embedding 원본
    object_id, object_name의 개수가 일치하지 않는 문제 -> 동일 id에 이름 두 개씩 들어가 있는 경우 발견
    -> 이름을 합침 '''

def adjColumn_kv(imgCount):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        dict = {}
        # dict 생성(obj_id : obj_name)
        for i in range(imgCount):
            imageDescriptions = data[i]["objects"]
            for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
                obj_id = imageDescriptions[j]['object_id']
                if (len(imageDescriptions[j]['names']) != 1):
                    wholeName = str()
                    for i in range(len(imageDescriptions[j]['names'])):
                        wholeName += imageDescriptions[j]['names'][i]
                    lista = []
                    lista.append(wholeName.replace(' ', ''))
                    obj_name = lista
                else:
                    obj_name = imageDescriptions[j]['names']

                dict[obj_id] = obj_name
            #  print(obj_id)
            #  print(obj_name)

        keys = sorted(dict)
        val = []

        for i in keys:
            if (type(dict[i]) == str):
                val += (dict[i])

            val += dict[i]
        return keys, val





''' 
feature matrix 2안 
scene graph에서 object-predicate-subject를 scenetence로 묶어서 임베딩 
-> 질문 : 이때 각 단어에 대한 임베딩은 어케 구할건지? 
    일일이 비교해서 구해야 하는지? 
    word가 아니고 phrase인 경우에는? 
    padding?
    '''


