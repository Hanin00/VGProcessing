import json

obj_id = 1546458  # computer
# obj_id = 1546459 #tower
with open('../data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    object = []

    imageDescriptions = data[0]["relationships"]
    print(("Scene : ", imageDescriptions))


with open('../data/region_graphs.json') as file:  # open json file
    data = json.load(file)
    object = []

    imageDescriptions = data[0]["regions"]
    print("region : ", imageDescriptions)
