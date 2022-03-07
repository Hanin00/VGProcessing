import util as ut


def main() :
    img_id = 1
    img_cnt = 1000
    adjColumn = ut.adjColumn(img_cnt)

    df_adj, adjMatrix = ut.createAdj(img_id, adjColumn)
    print("df_adj : ", df_adj)
    print("adjMatrix : ", adjMatrix)

    jsonpath= './data/region_descriptions.json'
    xlxspath= './data/image_regions.xlsx'
    ut.jsontoxml(img_cnt, jsonpath, xlxspath)











# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()



