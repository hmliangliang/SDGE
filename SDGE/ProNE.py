# -*- coding: utf-8 -*-#
import numpy as np
import networkx as nx

import scipy.sparse
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv

from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd
from warnings import simplefilter
simplefilter(action='ignore',category=FutureWarning)


class ProNE():
	def __init__(self, graph, Z, dimension):
		self.G = graph
		self.dimension = dimension
		self.node_number = self.G.number_of_nodes()
		self.A = nx.adjacency_matrix(self.G)
		self.a = Z

	def get_embedding_dense(self, matrix):
		# get dense embedding via SVD
		dimension = self.dimension
		U, s, Vh = linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
		U = np.array(U)
		U = U[:, :dimension]
		s = s[:dimension]
		s = np.sqrt(s)
		U = U * s
		U = preprocessing.normalize(U, "l2")
		return U

	def chebyshev_gaussian(self, order=10, mu=0.5, s=0.5):
		# NE Enhancement via Spectral Propagation
		#print('Chebyshev Series -----------------')
		A = self.A
		a = self.a
		if order == 1:
			return a
		A = sp.eye(self.node_number) + A
		DA = preprocessing.normalize(A, norm='l1')
		L = sp.eye(self.node_number) - DA
		M = L - mu * sp.eye(self.node_number)
		Lx0 = a
		Lx1 = M.dot(a)
		Lx1 = 0.5 * M.dot(Lx1) - a
		conv = iv(0, s) * Lx0
		conv -= 2 * iv(1, s) * Lx1
		for i in range(2, order):
			Lx2 = M.dot(Lx1)
			Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
			#         Lx2 = 2*L.dot(Lx1) - Lx0
			if i % 2 == 0:
				conv += 2 * iv(i, s) * Lx2
			else:
				conv -= 2 * iv(i, s) * Lx2
			Lx0 = Lx1
			Lx1 = Lx2
			del Lx2
			#print('Bessell time', i, time.time() - t1)
		mm = A.dot(a - conv)
		emb = self.get_embedding_dense(mm)
		return emb
