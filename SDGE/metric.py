# -*- coding: utf-8 -*-#
# Author:       Liangliang
# Date:         2020\9\18 0018 15:33:33
# File:         metric.py
# Software:     PyCharm
#------------------------------------
import math
from sklearn.metrics import normalized_mutual_info_score


def metric(y, y_pre):
    '''
    本代码主要是计算聚类的各种度量指标,该方法计算的时间复杂度为O(n^2).
    计算方法参照于'谢娟英. 无监督学习方法及其应用[M]. 电子工业出版社, 2016: 72-80.'
    y: array类型，真实的类标签, 格式1*n的一维数组,y[i]代表第i个样本的类标签
    y_pre: array类型，预测的类标签即聚类的类标签，格式1*n的一维数组,y[i]代表第i个样本的聚类的类标签
    return: R, J, ARI, FM, F1, Hubert, Phi, K, RT, NMI
    '''
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    n = len(y)
    N = n*(n-1)/2
    for i in range(n-1):
        for j in range(i+1,n):
            if y[i] == y[j]:#样本i与样本j处于同一个类中
                if y_pre[i] == y_pre[j]:#样本i与样本j处于同一个类簇中
                    TP = TP + 1
                else:
                    FN = FN + 1
            else:#样本i与样本j处于不同一个类中
                if y_pre[i] == y_pre[j]:#样本i与样本j处于同一个类簇中
                    FP = FP + 1
                else:
                    TN = TN + 1
    '''计算各种度量指标,未特殊说明,其值范围在[0,1],其值越大越好'''
    e = 1e-10#防止除数出现为0的情形
    R = (TP + TN)/N #Rand系数
    J = TP/(TP+FN+FP+e) #Jaccard系数
    ARI = 2*(TP*TN-FN*FP)/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)+e) #ARI指数
    FM = TP/(math.sqrt((TP+FN)*(TP+FP))+e)#Folkes and Mallows指标
    prcision = TP/(TP+FP+e)
    recall = TP/(TP+FN+e)
    F1 = 2*prcision*recall/(prcision+recall+e)#计算F-measure指标
    Hubert = (N*TP-(TP+FN)*(TP+FP))/math.sqrt((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP)+e)#Hubert指数，范围[-1,1]，其值越大越好
    Phi = (TP*TN-FN*FP)/math.sqrt((TP+FN)*(TP+FP)*(FN+TN)*(FP+TN)+e)#Phi指数，范围[-1,1]，其值越大越好
    K = 1/2*(TP/(TP+FP+e)+TP/(TP+FN+e))#Kulczynski指标
    RT = (TP+TN)/(TP+TN+2*(FN+FP)+e)#Rogers-Tanimoto指标
    NMI = normalized_mutual_info_score(y, y_pre)#标准互信息
    return R, J, ARI, FM, F1, Hubert, Phi, K, RT, NMI