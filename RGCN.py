def masked_loss(out, label, mask):
    '''
    loss function으로 이용되는 함수이다. mask 벡터의 원소가 True면 1 False면 0이기 때문에 True인 부분의 loss만을 가중치를 주어 계산하고,
    0인 부분, 즉 False인 부분은 masking이 된 것으로 판단하여 loss 계산에서 제외한다.
    '''
    loss = F.cross_entropy(out, label, reduction='none')  # 우선 레이블이 잇는 데이터의 아웃풋과 레이블의 크로스엔트로피로스를 구한다.
    mask = mask.float()  # 0,1 (False, True의 int형)로 이루어진 벡터로 되어있는 mask를 float으로 바꿔준다.
    mask = mask / mask.mean()  # 벡터의 각 값을 mask의 평균으로 나눈다.
    loss *= mask  # mask 벡터에 loss를 곱한다.
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    '''
    accuracy를 구하는 함수이다. 위와 마찬가지로 마스킹이 되지 않은 부분, 즉 마스크 벡터가 True인 부분의 accuracy를 구한다. 
    '''
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc


def sparse_dropout(x, rate, noise_shape):
    '''
    drop-out을 시행하는 함수이다. input matrix와 dropout의 비율과 0이 아닌 값을 가지는 원소의 개수를 argument로 받는다.
    '''

    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices()  # [2, 49216]
    v = x._values()  # [49216]

    # [2, 49216] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1. / (1 - rate))

    return out


class GraphConvolution(nn.Module):  # graph convolution을 실행할 함수를 만든다.

    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation=F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))  # weight parameter matrix를 차원에 맞춰 랜덤한 수를 넣어 만든다.
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))  # bias parameter matrix를 차원에 맞춰 0을 패딩하여 만든다.

    def forward(self, inputs):

        '''
        forward()는 모델이 학습데이터를 입력받아서 forward propagation을 진행시키는 함수이고, 반드시 forward 라는 이름의 함수이어야 한다.
        forward()는 __call__의 역할을 수행하기 때문에, 인스턴스를 만들면 바로 실행된다. model.forward(inputs)가 아닌 model(inputs)로 forward를 진행한다.
        '''

        x, support = inputs  # support는 classification을 할 때 중심이 되는 벡터

        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless:  # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)  # sparse matrix 형태의 input * weight
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias  # input * weight + bias

        return self.activation(out), support


class GCN(nn.Module):

    def __init__(self, input_dim, output_dim, num_features_nonzero):
        super(GCN, self).__init__()

        self.input_dim = input_dim  # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)

        self.layers = nn.Sequential(GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=True),

                                    GraphConvolution(args.hidden, output_dim, num_features_nonzero,
                                                     activation=F.relu,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False),

                                    )

    def forward(self, inputs):

        x, support = inputs

        x = self.layers((x, support))
        '''
        inputs라는 하나의 arg를 받기때문에 x, support를 튜플로 묶어서 input으로 넣고 GraphConvoltion class 의 forward를 실행시킨다.
        '''
        return x

    def l2_loss(self):  # weight^2을 regularization으로 준다.

        layer = self.layers.children()  # gives an iterator over the layers in network model.
        layer = next(iter(layer))  # 레이어를 iterate하게 꺼낸다.

        loss = None

        for p in layer.parameters():  # 레이어의 weight를 제곱하여 loss에 더한다.
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss
