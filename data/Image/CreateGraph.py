# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2020\10\13 0013 21:18:18
# File:         CreateGraph.py
# Software:     PyCharm
#------------------------------------
import numpy as np
import scipy.io as scio
from sklearn.neighbors import NearestNeighbors

class CreateGraph():
    def __init__(self,address,name,k,kind='unweighted'):
        self.address = address  #The path of the file
        self.name = name #The name of the file
        self.k = k #The number of the nearest neighbors
        self.kind = kind #kind='weighted': wighted graph  or unwighted graph
    def ReadLabel(self):
        data = scio.loadmat(self.address+'/'+self.name+'.mat')
        data = np.array(data[self.name],dtype = float)
        data = data[0:3500,data.shape[1]-1]-1
        f = open(self.address+'/'+self.name+'-labels.txt','w')
        for i in range(len(data)):
            f.write(str(int(data[i]))+'\n')
        print('Read Label is finishing!')
        f.close()
    def ReadData(self):
        data = scio.loadmat(self.address+'/'+self.name+'.mat')
        data = np.array(data[self.name], dtype=float)
        data = data[0:3500,0:data.shape[1]-1]
        np.savetxt(self.address+'/'+self.name+'.txt', data, fmt='%f', delimiter=' ')
        print('Read data is finishing!')
        return data
    def Create(self):
        data = self.ReadData()
        nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='kd_tree').fit(data)
        distances, indices = nbrs.kneighbors(data)
        f = open(self.address + '/' + self.name + '-graph.txt', 'w')
        for i in range(indices.shape[0]):
            for j in range(1,indices.shape[1]):
                s = ''
                s = str(int(i))+' '+str(int(indices[i,j]))+'\n'
                f.write(s)
        f.close()
        print('Write operation is finishing!')

if __name__ == '__main__':
    address = 'C:/Users/Administrator/Desktop'
    name = 'Image'
    k = 10
    g = CreateGraph(address,name,k)
    g.ReadLabel()
    g.Create()
