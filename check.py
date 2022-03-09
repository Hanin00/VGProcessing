import util as ut
# 단어 빈도수 기반 임베딩을 하고 있음.. 되는거 하자 되는거.

img_id = 1
img_cnt = 1000
adjColumn, xWords = ut.adjColumn_kv(img_cnt)

# for i in range(len(xWords)) :
#     xWords[i] = xWords[i].replace(' ','')

a = []
a.append(xWords)

from gensim.models import Word2Vec
#model = Word2Vec(sentences=a, vector_size=10, window=1, min_count=5, workers=4, sg=0)

model = Word2Vec(a,window = 5,min_count=2,sg=1,iter=10000)
print(model.wv['shade'])
print(model.wv['walksign'])


'''
for i in range(len(a[0])) :
    print("a : ",a[0][i],"   model.wv[a[0][i]] : " ,model.wv[a[0][i]])'''

'''all_vectors = []
for i in a[0]:
    vectors = model.wv[i]
    all_vectors.append(vectors)

print(len(all_vectors))'''