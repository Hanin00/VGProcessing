import createAdj as cAdj
import jsontoxml as jtx

img_id = 1
img_cnt = 1000
adjColumn = cAdj.adjColumn(img_cnt)

df_adj, adjMatrix = cAdj.createAdj(img_id, adjColumn)
print(df_adj)
print(adjMatrix)


jtx.