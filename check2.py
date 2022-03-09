import json
from openpyxl import Workbook

imgCnt = 1000
jsonpath = './data/scene_graphs.json'
xlsxpath = './data/scene_sentence.xlsx'

with open(jsonpath) as file:  # open json file
    data = json.load(file)
    wb = Workbook()  # create xlsx file
    ws = wb.active  # create xlsx sheet
    ws.append(['image_id', 'scene_sentence'])

    phrase = []

    q = 0
    for i in data:
        if q == imgCnt:
            break
        #image_id
        imgId = i.get('image_id')
        #objName, subjectName
        object = i.get('objects')
        objName = []
        for j in object :
                objName.append(object[j].get('name'))
        print("len(objName) : ", len(objName))
        # predicate
        k = 0
        for j in object:
            if k == 2:
                break
            phrase.append(j.get('phrase'))
            k += 1



        sentences = ','.join(phrase)
        ws.append([imgId, sentences])
        phrase = []
        q += 1
    wb.save(xlsxpath)
