import json
from openpyxl import Workbook


"""
    Y data 생성을 위해 image에 대한 text description을 이미지 별로 모음
"""

#jsonpath : './data/region_descriptions.json'
#xlxspath : './data/image_regions.xlsx'
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