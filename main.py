import util as ut
import YEmbedding as yed
import pandas as pd

import sys


def main():
    img_id = 1
    img_cnt = 1000
    adjColumn, xWords = ut.adjColumn_kv(img_cnt)
    # 개수 동일, 순서는 X sorted로 정렬하면 list로 바뀌더라고..? for문으로 호출하면 될 듯..?헐랭 해결해따

    '''개수 확인'''
    df_adj, adjMatrix = ut.createAdj(img_id, adjColumn)
    print("adjMatrix : ", adjMatrix)

    jsonpath= './data/region_descriptions.json'
    xlxspath= './data/image_regions.xlsx'
    ut.jsontoxml(img_cnt, jsonpath, xlxspath)


    yed.YEmbedding(xlxspath)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
