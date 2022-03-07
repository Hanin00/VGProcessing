import json
from visual_genome import api as vg
import pandas as pd
from openpyxl import Workbook

with open('C:/Users/Haeun/PycharmProjects/VGpreprocessing/data/objects.json', mode='r', encoding='utf-8') as file:  # objects.json locally
    data = json.load(file)
    wb = Workbook() # create xlsx file
    ws = wb.active  # create xlsx sheet
    ws.append(['image_id','region_sentences'])

regions = []
images = []
for i in range(len(data)):
    region_sentences = []  # descriptions for areas of the chosen image
    image_id = data[i]['image_id']
    try:
        region = vg.get_region_descriptions_of_image(id=image_id)
        for j in region:
            region_sentences.append(j.phrase.lower())
        if region_sentences:
            images.append(image_id)
            regions.append(region_sentences)
    except IndexError:
        continue


ws.append([images,regions])
wb.save('./data/image_regions2.xlsx')
images_regions = pd.DataFrame(list(zip(images, regions)), columns=['Image_id', 'region_sentences'])


print(images_regions[0])

