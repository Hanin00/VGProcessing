import json
import pprint

# 한 이미지당 모든 object 가져와서 리스트로 만들고 embedding(word2vec)
# 리스트로 만들고
# 인접행렬(relationship) 과 feature(object id) 행렬 만들기 -> 
# object명 set으로 안겹치게 column 명... 걍 object id 중에서 cite 많이 된 순서로

with open('./data/scene_graphs.json') as file:  # open json file
    data = json.load(file)
    pp = pprint.PrettyPrinter(indent=4)
    print(pp.pprint(data[0]))


