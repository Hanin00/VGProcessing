import json


imgCount = 1000
''' object_id, object_name의 개수가 일치하지 않는 문제 -> 동일 id에 이름 두 개씩 들어가 있는 경우 발견
    -> 이름을 합침 '''



with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    dict = {}
    # dict 생성(obj_id : obj_name)
    for i in range(imgCount):
        imageDescriptions = data[i]["objects"]
        for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            obj_id = imageDescriptions[j]['object_id']
            if(len(imageDescriptions[j]['names'])!=1) :
                wholeName= str()
                for i in range(len(imageDescriptions[j]['names'])) :
                        wholeName+= imageDescriptions[j]['names'][i]
                lista = []
                lista.append(wholeName.replace(' ',''))
                obj_name = lista
            else :
                obj_name = imageDescriptions[j]['names']

            dict[obj_id] = obj_name
          #  print(obj_id)
          #  print(obj_name)

    keys = sorted(dict)
    val = []

    for i in keys:
        if(type(dict[i]) == str) :
            val += (dict[i])

        val += dict[i]
        print(dict[i])
        print(type(dict[i]))

    print("keys : ", len(keys))
    print("val : ",len(val))