import json
from openpyxl import Workbook

with open('./data/region_descriptions.json') as file:   # open json file
    data = json.load(file)
    wb = Workbook() # create xlsx file
    ws = wb.active  # create xlsx sheet
    ws.append(['image_id','region_sentences'])

    phrase = []

    q = 0
    for i in data:
        if q==700 :
            break
        regions = i.get('regions')
        imgId = regions[0].get('image_id')
        k = 0
        for j in regions:
            if k==7:
                break
            phrase.append(j.get('phrase'))
            k+=1
        sentences = ','.join(phrase)
        ws.append([imgId, sentences])
        phrase = []
        q+=1
    wb.save('./data/image_regions.xlsx')


'''
    phrase = []
    k = 0
    for i in data:
        regions = i.get('regions')
        imgId = regions[0].get('image_id')
        for j in regions:
            phrase.append(j.get('phrase'))
        sentences = ','.join(phrase)
        ws.append([imgId, sentences])
        phrase = []
    wb.save('./data/image_regions.xlsx')
'''