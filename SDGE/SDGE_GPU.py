# -*- encoding: utf-8 -*-
"""
@File    : SDGE.py
@Time    : 9/5/20 9:09 PM
@Author  : Liangliang
@Software: PyCharm
"""
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import metric
from tqdm import tqdm
import math
from sklearn.cluster import KMeans
from sklearn import preprocessing
import ProNE
import DynamicReLU
from sklearn.manifold import SpectralEmbedding
from warnings import simplefilter
simplefilter(action='ignore',category=FutureWarning)
import os
os.environ['CUDA_VISIBLE_DEVICES']= '1'

class Parameters():#设置SDGE算法的参数
    def __init__(self,beta=1,gamma=1,fun='sotmax',d=64,k=3,layers=[200,170,140,100],cat='CAT'):
        self.beta = beta
        self.gamma = gamma
        self.fun = fun # 'softmax' or 'k-means'
        self.d = d #The dimension of low embedding vectors
        self.k = k #The number of community
        self.layers=layers #the numbers of GCN hidden nodes in each layer
        self.cat = cat #输出的拼接方式 sum or cat

class Net():#定义图卷积神经网络结构
    def __init__(self,parameters,graph,r=2,X=None):
        self.parameters = parameters
        self.graph = graph
        self.r = r
        if X == None:
            self.X = torch.eye(self.graph.number_of_nodes()).cuda()
        else:
            self.X = X.cuda()
        self.W1 = torch.rand(self.X.shape[1], self.parameters.layers[0], requires_grad=True)
        self.W2 = torch.rand(self.parameters.layers[0], self.parameters.layers[1], requires_grad=True)
        self.W3 = torch.rand(self.parameters.layers[1], self.parameters.layers[2], requires_grad=True)
        self.W4 = torch.rand(self.parameters.layers[2], self.parameters.layers[3], requires_grad=True)

    def forward(self):
        A = np.array(nx.adjacency_matrix(self.graph).todense())
        A = np.power(A,self.r)
        A = torch.tensor(A).cuda()
        A = A + torch.eye(A.shape[0]).cuda()
        D = torch.diag(torch.sum(A, dim=1)).sqrt().float().cuda()
        # 三隐含层连接权值
        H = self.X.float()
        # 三层图卷积操作
        H = D.mm(A.mm(D.mm(H.mm(self.W1.cuda()))))  # 第1层卷积层
        model = DynamicReLU.DyReLUB(channels=H.shape[0], conv_type='1d').cuda()
        H = model(H.reshape(1,H.shape[0],H.shape[1]))[0]
        Normalization = nn.BatchNorm1d(H.shape[1],affine=True).cuda()
        H = Normalization(H)
        H = D.mm(A.mm(D.mm(H.mm(self.W2.cuda()))))  # 第2层卷积层
        model = DynamicReLU.DyReLUB(channels=H.shape[0], conv_type='1d').cuda()
        H = model(H.reshape(1, H.shape[0], H.shape[1]))[0]
        Normalization = nn.BatchNorm1d(H.shape[1], affine=True).cuda()
        H = Normalization(H)
        H = D.mm(A.mm(D.mm(H.mm(self.W3.cuda()))))  # 第3层卷积层
        model = DynamicReLU.DyReLUB(channels=H.shape[0], conv_type='1d').cuda()
        H = model(H.reshape(1, H.shape[0], H.shape[1]))[0]
        H = D.mm(A.mm(D.mm(H.mm(self.W4.cuda()))))  # 输出层
        Normalization = nn.BatchNorm1d(H.shape[1], affine=True).cuda()
        H = Normalization(H)
        return H

class NetF(nn.Module):
    def __init__(self,num1,num2):
        self.num1 = num1
        self.num2 = num2
        super(NetF, self).__init__()
        self.fc1 = nn.Linear(self.num1, 80)
        self.fc2 = nn.Linear(80, self.num2)
    def forward(self,data):
        data = self.fc1(data)
        data = torch.sigmoid(data)
        data = self.fc2(data)
        return data

class SDGE():
    def __init__(self,graph,X,parameters,num):
        self.parameters = parameters
        self.graph = graph
        self.num = num
        #self.feature = torch.tensor(SpectralEmbedding(n_components=parameters.d).fit_transform(np.array(nx.adjacency_matrix(graph).todense())),dtype=torch.float32).cuda()
        if X == None:
            self.X = torch.eye(self.graph.number_of_nodes()).cuda()
        else:
            self.X = X.cuda()

    def Modularity(self,pred):  # 计算模块度
        modularity = 0
        edges = nx.edges(self.graph)
        m = self.graph.number_of_edges()
        for e in edges:
            if pred[e[0]] == pred[e[1]]:
                    modularity = modularity + 1 / (2 * m) * (1 - (self.graph.degree(e[0]) * self.graph.degree(e[1])) / (2 * m))
        return modularity

    def weight(self, H):  #由模块度计算每个输出的权值
        w = []
        for h in H:
            pred = KMeans(n_clusters=self.parameters.k).fit_predict(h)
            w.append(self.Modularity(pred))
        w = F.softmax(torch.tensor(np.array([w]), dtype=float), dim=1)
        return w[0]

    def noise(self, signal):  # 生成高斯分布噪声 https://blog.csdn.net/sinat_24259567/article/details/93889547
        SNR = 5
        noise = torch.randn(signal.shape[0], signal.shape[1]).cuda()  # 产生N(0,1)噪声数据
        noise = noise - torch.mean(noise)  # 均值为0
        signal_power = torch.norm(signal, 'fro') ** 2 / (signal.shape[0] * signal.shape[1])  # 此处是信号的std**2
        noise_variance = signal_power / torch.pow(torch.tensor(10), (SNR / 10))  # 此处是噪声的std**2
        noise = (torch.sqrt(noise_variance) / torch.std(noise)) * noise  ##此处是噪声的std**2
        signal_noise = noise + signal
        return signal_noise

    def sample(self, i, k):
        s = []
        neigbors = [_ for _ in nx.all_neighbors(self.graph, i)]
        while len(s) < k:
            j = np.random.randint(0, self.graph.number_of_nodes(), (1, 1))[0][0]
            if j not in neigbors:
                p = 1 / (1 + math.exp(-math.pow(self.graph.degree(j), 3 / 4)))  # 原理见LINE算法
                g = np.random.rand(1, 1)[0][0]
                if g >= p:  # accept the sample
                    s.append(j)
        return s

    def LossFunction(self, Z):
        A = torch.tensor(np.array(nx.adjacency_matrix(self.graph).todense())).cuda()
        Xn = self.noise(Z)
        # 自监督学习损失
        n = self.graph.number_of_nodes()
        Ls = 0
        tau = 2
        k = 20  # 负样本采样数
        for i in range(n):
            x = Z[i, :]
            xs = Xn[i, :]
            Ls = Ls - torch.log(1 / (1 + torch.exp(-torch.dot(x, xs)/tau)))
            samples = self.sample(i, k)
            for s in samples:
                Ls = Ls + 1 / len(samples) * torch.log(1 / (1 + torch.exp(-torch.dot(x, Z[s, :])/tau)))
        # 图的结构信息与属性信息约束
        Lsa = 0
        Lsa = Lsa + torch.norm(torch.mm(Z, Z.transpose(0,1)) - A, 'fro') ** 2
        Lsa = Lsa + torch.norm(torch.mm(Z, Z.transpose(0,1)) - torch.mm(self.X, self.X.transpose(0,1)), 'fro') ** 2
        del A
        # 图的正则化约束
        L = torch.tensor(nx.laplacian_matrix(self.graph).todense()).float().cuda()
        Lr = torch.trace(torch.mm(Z.transpose(0,1), torch.mm(L, Z)))
        return Ls/(k*n) + self.parameters.beta * Lsa/n**2 + self.parameters.gamma * Lr/n

    def fit(self):
        H = []
        net1 = Net(self.parameters, self.graph, 1, self.X)
        h1 = net1.forward()
        H.append(h1.detach().cpu().numpy())
        net2 = Net(self.parameters, self.graph, 2, self.X)
        h2 = net2.forward()
        H.append(h2.detach().cpu().numpy())
        net3 = Net(self.parameters, self.graph, 3, self.X)
        h3 = net3.forward()
        H.append(h3.detach().cpu().numpy())
        net4 = Net(self.parameters, self.graph, 4, self.X)
        h4 = net4.forward()
        H.append(h4.detach().cpu().numpy())
        w = self.weight(H)
        if self.parameters.cat == 'sum':  # 加权求和
            H = w[0] * h1 + w[1] * h2 + w[2] * h3 + w[3] * h4
        else:  # 拼接各个GCN的输出结果
            H = torch.tensor([]).cuda()
            H = torch.cat((H, w[0] * h1), 1).cuda()
            H = torch.cat((H, w[1] * h2), 1).cuda()
            H = torch.cat((H, w[2] * h3), 1).cuda()
            H = torch.cat((H, w[3] * h4), 1).cuda()
        if self.X.shape[0] == self.X.shape[1]:
            if (self.X == torch.eye(X.shape[1]).cuda()).any() == False:  # 判断是否是单位阵
                H = torch.cat((H, self.X), 1).cuda()
        else:
            H = torch.cat((H, self.X), 1).cuda()
        if self.parameters.fun == 'softmax':  # 采用端到端方式,直接获得社团划分结果
            net5 = NetF(H.shape[1],self.parameters.k).cuda()
        else:  # 使用k-means算法获得类簇的划分
            net5 = NetF(H.shape[1],self.parameters.d).cuda()
        Normalization = nn.BatchNorm1d(H.shape[1], affine=True).cuda()
        H = Normalization(H)
        Z = net5.forward(H)
        lr = 0.0001
        optimzer = torch.optim.Adam(net5.parameters(), lr=lr, betas=(0.9, 0.999))
        Loss_list = []
        print('\n模型正在训练中....................')
        for epoch in range(250):
            loss = self.LossFunction(Z)
            Loss_list.append(loss.item())
            optimzer.zero_grad() # 清除梯度
            loss.backward()
            # 采样梯度下降法更新GCN参数
            net1.W1 = net1.W1 - lr * net1.W1.grad
            net1.W1 = net1.W1.clone().detach().requires_grad_(True)
            net1.W2 = net1.W2 - lr * net1.W2.grad
            net1.W2 = net1.W2.clone().detach().requires_grad_(True)
            net1.W3 = net1.W3 - lr * net1.W3.grad
            net1.W3 = net1.W3.clone().detach().requires_grad_(True)
            net1.W4 = net1.W4 - lr * net1.W4.grad
            net1.W4 = net1.W4.clone().detach().requires_grad_(True)
            net2.W1 = net2.W1 - lr * net2.W1.grad
            net2.W1 = net2.W1.clone().detach().requires_grad_(True)
            net2.W2 = net2.W2 - lr * net2.W2.grad
            net2.W2 = net2.W2.clone().detach().requires_grad_(True)
            net2.W3 = net2.W3 - lr * net2.W3.grad
            net2.W3 = net2.W3.clone().detach().requires_grad_(True)
            net2.W4 = net2.W4 - lr * net2.W4.grad
            net2.W4 = net2.W4.clone().detach().requires_grad_(True)
            net3.W1 = net3.W1 - lr * net3.W1.grad
            net3.W1 = net3.W1.clone().detach().requires_grad_(True)
            net3.W2 = net3.W2 - lr * net3.W2.grad
            net3.W2 = net3.W2.clone().detach().requires_grad_(True)
            net3.W3 = net3.W3 - lr * net3.W3.grad
            net3.W3 = net3.W3.clone().detach().requires_grad_(True)
            net3.W4 = net3.W4 - lr * net3.W4.grad
            net3.W4 = net3.W4.clone().detach().requires_grad_(True)
            net4.W1 = net4.W1 - lr * net4.W1.grad
            net4.W1 = net4.W1.clone().detach().requires_grad_(True)
            net4.W2 = net4.W2 - lr * net4.W2.grad
            net4.W2 = net4.W2.clone().detach().requires_grad_(True)
            net4.W3 = net4.W3 - lr * net4.W3.grad
            net4.W3 = net4.W3.clone().detach().requires_grad_(True)
            net4.W4 = net4.W4 - lr * net4.W4.grad
            net4.W4 = net4.W4.clone().detach().requires_grad_(True)
            optimzer.step()
            # 重新计算低维嵌入
            H = []
            h1 = net1.forward()
            H.append(h1.detach().cpu().numpy())
            h2 = net2.forward()
            H.append(h2.detach().cpu().numpy())
            h3 = net3.forward()
            H.append(h3.detach().cpu().numpy())
            h4 = net4.forward()
            H.append(h4.detach().cpu().numpy())
            w = self.weight(H)
            if self.parameters.cat == 'sum':  # 加权求和
                H = w[0] * h1 + w[1] * h2 + w[2] * h3 + w[3] * h4
            else:  # 拼接各个GCN的输出结果
                H = torch.tensor([]).cuda()
                H = torch.cat((H, w[0] * h1), 1).cuda()
                H = torch.cat((H, w[1] * h2), 1).cuda()
                H = torch.cat((H, w[2] * h3), 1).cuda()
                H = torch.cat((H, w[3] * h4), 1).cuda()
            if self.X.shape[0] == self.X.shape[1]:
                if (self.X == torch.eye(self.X.shape[1]).cuda()).any() == False:  # 判断是否是单位阵
                    H = torch.cat((H, self.X), 1).cuda()
            else:
                H = torch.cat((H, self.X), 1).cuda()
            Normalization = nn.BatchNorm1d(H.shape[1], affine=True).cuda()
            H = Normalization(H)
            Z = net5.forward(H)
            print('第',self.num,'次运行算法中的第', epoch + 1, '波训练的损失函数值loss=', loss.detach().cpu().numpy())
        print('模型训练过程已结束！')
        file = open('acm-loss-cat.txt', 'w')
        file.write(str(Loss_list))
        file.close()
        print('文件写入完成!')
        Z = Z.detach().cpu().numpy()
        Z = ProNE.ProNE(self.graph, Z, self.parameters.d).chebyshev_gaussian()# 使用ProNE算法对嵌入结果进行增强
        return Z

    def predict(self,Z):#'softmax' or 'k-means'
        if self.parameters.fun == 'softmax':
            Z = F.softmax(torch.tensor(Z), dim=1)
            pred = torch.argmax(Z, dim=1).numpy()
        else:
            Z = F.softmax(torch.tensor(Z), dim=1).numpy()
            pred = KMeans(n_clusters=self.parameters.k).fit_predict(Z)
        return pred

if __name__ == '__main__':
    start = time.time()
    labels = np.loadtxt('../data/acm/acm-labels.txt',dtype=int)
    data = np.loadtxt('../data/acm/acm-graph.txt',dtype=int).tolist()
    flag = int(input('Is graph an attribute graph? (1.YES 2.NO):'))
    if flag == 1:
        X = torch.tensor(preprocessing.scale(np.loadtxt('../data/acm/acm.txt',dtype=float)),dtype=torch.float32)
    else:
        X = torch.eye(len(labels),dtype=torch.float32)
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(labels))])
    g.add_edges_from(data)
    del data
    k_clusters = len(np.unique(labels))
    parameters = Parameters(beta=1, gamma=1, fun='softmax', d=64, k=k_clusters,layers=[200,170,140,100],cat='CAT')# Set the parameters of SDGE
    num = 1
    RR = []
    JJ = []
    ARII = []
    FMM = []
    FF1 = []
    Hubertt = []
    Phii = []
    KK = []
    RTT = []
    NMII = []
    print('The implementation is running on', (chr(0x266B) + ' ') * 45)
    for t in tqdm(range(num)):
        model = SDGE(g, X, parameters,t+1)
        Z = model.fit()
        y_pred = model.predict(Z)
        R, J, ARI, FM, F1, Hubert, Phi, K, RT, NMI = metric.metric(labels, y_pred)
        RR.append(R)
        JJ.append(J)
        ARII.append(ARI)
        FMM.append(FM)
        FF1.append(F1)
        Hubertt.append(Hubert)
        Phii.append(Phi)
        KK.append(K)
        RTT.append(RT)
        NMII.append(NMI)
    print('R=', round(np.mean(RR), 4), '$\pm$', round(np.std(RR), 4))
    print('J=', round(np.mean(JJ), 4), '$\pm$', round(np.std(JJ), 4))
    print('ARI=', round(np.mean(ARII), 4), '$\pm$', round(np.std(ARII), 4))
    print('FM=', round(np.mean(FMM), 4), '$\pm$', round(np.std(FMM), 4))
    print('F1=', round(np.mean(FF1), 4), '$\pm$', round(np.std(FF1), 4))
    print('Hubert=', round(np.mean(Hubertt), 4), '$\pm$', round(np.std(Hubertt), 4))
    print('Phi=', round(np.mean(Phii), 4), '$\pm$', round(np.std(Phii), 4))
    print('K=', round(np.mean(KK), 4), '$\pm$', round(np.std(KK), 4))
    print('RT=', round(np.mean(RTT), 4), '$\pm$', round(np.std(RTT), 4))
    print('NMI=', round(np.mean(NMII), 4), '$\pm$', round(np.std(NMII), 4))
    end = time.time()
    print('The time cost is',(end - start)/num)