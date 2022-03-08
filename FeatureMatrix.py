import json
from openpyxl import Workbook
import util as ut

img_id = 1
img_cnt = 1000

'''1000개 obj에 대한 임베딩 값'''

adjColumn, xWords = ut.adjColumn_kv(img_cnt)

print(len(xWords))