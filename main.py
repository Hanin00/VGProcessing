import util as ut
import YEmbedding as yed
import numpy as np
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tnrange
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from scipy.sparse import eye
from pathlib import Path
from functools import partial




'''
    csv 파일로 만들어야 하는 것(이미지 천 개에 대한 데이터들) : region_descriptiong> image_regions
        columns Name(31310) 
        featuremap(31310)
        이미지 1000개에 대한 adjMatrix(train 용) 
        > 근데 어차피 다 쓸 거 아니니까 상관 없지 않나 싶기두.. 일부만 할거면.. 걍 만들어놓고 갖다 쓰는게 편하겠지..?
'''




def main():
    img_id = 1
    img_cnt = 1000

    #textEmbedding하기 위한 이미지별 phrase 모아서 xlsx로 변경
    jsonpath = './data/region_descriptions.json'
    xlxspath = './data/image_regions.xlsx'
    ut.jsontoxml(img_cnt, jsonpath, xlxspath)


    # obj에 대한 text embedding 값 -> 모든 objName에 대해 동일값 들어감.
    adjColumn, xWords = ut.adjColumn_kv(img_cnt)
    # X - AdjMatrix(numpy.ndarray, 31310x31310)
    df_adj, adjMatrix = ut.createAdj(img_id, adjColumn)
    print(type(adjMatrix))
    print('adjMatrix[0].size() : ',adjMatrix[0].size)
    print('adjMatrix[1].size() : ',adjMatrix[1].size)

    #X - Featuremap(list, [31310])
    featureEmbedding = ut.objNameEmbedding(xWords)
    print('featureEmbedding type : ',type(featureEmbedding))
    print('featureEmbedding length : ',len(featureEmbedding))


    #YEmbedding -  각 이미지 별 클러스터 값..? 이거 뭘로 Y 값을 만들어야 하지..?
    #원래 아이디어 : 각 이미지 별 클러스터 값, 동일한 지 파악해서 T, F 반환. 이거 1000x1000?
    #feature matrix는 동일하니까?
    #이미지1 X(adj) 값과 이미지1의 클러스터값 + 이미지2(adj)의 X값과 이미지 2의 클러스터값 train set으로 주고,
    #판별 할 때 클러스터 값 동일한 지 여부 파악하게 하면 될 것 같은데 어케하지..?

    idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]

    print(type(idCluster))




    #features = csr_matrix(paper_features_label[:, 1:-1], dtype=np.float32)
    features = adjMatrix
    labels = xWords
    #lbl2idx = {k: v for v, k in enumerate(sorted(np.unique(labels)))}
    #labels = [lbl2idx[e] for e in labels]
    #labels[:5]

    papers = paper_features_label[:, 0].astype(np.int32)
    # 걍 1000개 이미지에 대한 Adj 만들고, 위에 idcluster[]에서 호출해서 사용하는 방법으로 써야할 듯.






'''
    embedding_clustering = yed.YEmbedding(xlxspath)
    # YEmbedding
    idCluster = embedding_clustering[['image_id','cluster','distance_from_centroid']]

    features = csr_matrix(paper_features_label[:, 1:-1], dtype=np.float32)
    labels = paper_features_label[:, -1]
    lbl2idx = {k: v for v, k in enumerate(sorted(np.unique(labels)))}
    labels = [lbl2idx[e] for e in labels]
    labels[:5]

    n_labels = labels.max().item() + 1
    n_features = features.shape[1]
    n_labels, n_features

    #train/val set 나눔
    np.random.seed(34)
    n_train = 200
    n_val = 300
    n_test = len(featureEmbedding) - n_train - n_val
    idxs = np.random.permutation(len(featureEmbedding))
    idx_train = torch.LongTensor(idxs[:n_train])
    idx_val = torch.LongTensor(idxs[n_train:n_train + n_val])
    idx_test = torch.LongTensor(idxs[n_train + n_val:])

    torch.manual_seed(34)

    model = GCN(nfeat=featureEmbedding,
                nhid=20,  # hidden = 16
                nclass=n_labels,
                dropout=0.5)  # dropout = 0.5
    optimizer = optim.Adam(model.parameters(),
                           lr=0.001, weight_decay=5e-4)

    def step():
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = F.nll_loss(output[idx_train], labels[idx_train])
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        return loss.item(), acc

    def evaluate(idx):
        model.eval()
        output = model(features, adj)
        loss = F.nll_loss(output[idx], labels[idx])
        acc = accuracy(output[idx], labels[idx])

        return loss.item(), acc

    def accuracy(output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    epochs = 1000
    print_steps = 100
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    for i in tnrange(epochs):
        tl, ta = step()
        train_loss += [tl]
        train_acc += [ta]

        if ((i + 1) % print_steps) == 0 or i == 0:
            tl, ta = evaluate(idx_train)
            vl, va = evaluate(idx_val)
            val_loss += [vl]
            val_acc += [va]

            print(
                'Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'.format(i, tl, ta, vl, va))

'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
