#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import random

class optStruct:
	def __init__(self,dataMat,label,C,toler):
		"""
			eCache第一列是是否有效的标志位，第二列是E值
		"""
		self.dataMat=dataMat
		self.label=label
		self.C=C
		self.toler=toler
		self.m=np.shape(dataMat)[0]
		self.alphas=np.mat(np.zeros((self.m,1)))
		self.b=0
		self.eCache=np.mat(np.zeros((self.m,2)))				#误差项

def loadData():
	"""
		读取文件中的数据
	"""
	dataMat=[]
	label=[]
	with open("testSet.txt") as f:
		for line in f.readlines():
			lineArr=line.strip().split("\t")
			dataMat.append([float(lineArr[0]),float(lineArr[1])])
			label.append(int(lineArr[2]))
	return dataMat,label
		
def calE(oS,k):
	"""
		计算E值并返回
	"""
	fx=float(np.multiply(oS.alphas,oS.label).T*(oS.dataMat*oS.dataMat[k,:].T))+oS.b
	E=fx-float(oS.label[k])
	return E
	
def clipAlpha(alpha,L,H):
	"""
		用于调整alpha
	"""
	if alpha<L:
		alpha=L
	if alpha>H:
		alpha=H
	return alpha
	
def selectRandJ(i,m):
	"""
		随机选择另一个alpha
	"""
	j=i
	while j==i:
		j=int(random.uniform(0,m))
	return j
	
	
def selectJ(i,oS,Ei):
	"""
		用于选择第二个alpha值，或者说是内循环的alpha值
		选择合适的alpha值以便在每次优化中采用最大步长
	"""
	
	maxK=-1
	maxDelt=0
	Ej=0
	oS.eCache[i]=[1,Ei]											#设置E为有效的，即为它已经计算好了
	validEcacheList=np.nonzero(oS.eCache[:,0].A)[0]				#返回的是eCache第一项不为0的eCache项的下标
	if len(validEcacheList)>1:
		for k in validEcacheList:
			if k==i:
				continue
			Ek=calE(oS,k)
			deltE=abs(Ek-Ei)
			if deltE>maxDelt:
				maxDelt=deltE									#选择具有最大步长的j
				maxK=k
				Ej=Ek
		return maxK,Ej
	else:
		j=selectRandJ(i,oS.m)
		Ej=calE(oS,j)
		return j,Ej
		
		
def updateE(oS,k):
	"""
		计算误差值并写入缓存中
	"""
	Ek=calE(oS,k)
	oS.eCache[k]=[1,Ek]
	
def innerLoop(i,oS):
	"""
		优化例程，同时选择两个alpha进行优化，内层循环
		返回1的话说明同时对两个alp进行了优化
	"""
	Ei=calE(oS,i)												#计算i的误差
	if (oS.label[i]*Ei<-oS.toler and oS.alphas[i]<oS.C) or (oS.label[i]>oS.toler and oS.alphas[i]>0):
		j,Ej=selectJ(i,oS,Ei)									#第二个alpha选择的启发算法
		alphaIold=oS.alphas[i].copy()
		alphaJold=oS.alphas[j].copy()							#python使用引用的方式传递所有列表，需要分配新的内存
		if oS.label[i]!=oS.label[j]:
			L=max(0,oS.alphas[j]-oS.alphas[i])
			H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
		else:
			L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
			H=min(oS.C,oS.alphas[j]+oS.alphas[i])
		if L==H:
			#print "L=H"
			return 0
		eta=oS.dataMat[i,:]*oS.dataMat[i,:].T+oS.dataMat[j,:]*oS.dataMat[j,:].T-2.0*oS.dataMat[i,:]*oS.dataMat[j,:].T
		if eta<=0:
			print "eta<=0"
			return 0
		oS.alphas[j]+=oS.label[j]*(Ei-Ej)/eta
		oS.alphas[j]=clipAlpha(oS.alphas[j],L,H)				#更新j的误差
		updateE(oS,j)											#更新误差缓存
		if abs(oS.alphas[j]-alphaJold)<0.00001:					#判断alpha是否有轻微改变
			#print "alpha is not moving enough"
			return 0
		oS.alphas[i]+=oS.label[j]*oS.label[i]*(alphaJold-oS.alphas[j])			#更新i的alpha
		updateE(oS,i)											#更新误差缓存
		
		b1=oS.b-Ei-oS.label[i]*oS.dataMat[i,:]*oS.dataMat[i,:].T*(oS.alphas[i]-alphaIold)-oS.label[j]*oS.dataMat[j,:]*oS.dataMat[i,:].T*(oS.alphas[j]-alphaJold)
		b2=oS.b-Ej-oS.label[j]*oS.dataMat[j,:]*oS.dataMat[j,:].T*(oS.alphas[j]-alphaJold)-oS.label[i]*oS.dataMat[j,:]*oS.dataMat[i,:].T*(oS.alphas[i]-alphaIold)
		
		if oS.alphas[i]>0 and oS.alphas[i]<oS.C:				#如果b1
			oS.b=b1
		elif oS.alphas[j]>0 and oS.alphas[j]<oS.C:
			oS.b=b2
		else:
			oS.b=(b1+b2)/2.0
		return 1
	else:
		return 0
		
def SMO(dataMat,label,C,toler,maxIter):
	oS=optStruct(np.mat(dataMat),np.mat(label).transpose(),C,toler)
	iter=0
	entireSet=True
	alphaPairsChanged=0
	while (iter<maxIter) and ((alphaPairsChanged>0) or entireSet):
		alphaPairsChanged=0															#优化的alpha对数
		if entireSet:
			for i in range(oS.m):
				alphaPairsChanged+=innerLoop(i,oS)
			print "fullset: %d pairs changed"% alphaPairsChanged
			iter+=1
		else:
			nonBoundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<oS.C))[0]			#提取非边界的alpha值，不在0和C上面
			for i in nonBoundIs:
				alphaPairsChanged+=innerLoop(i,oS)
			print "nonBound: %d pairs changed"% alphaPairsChanged
			iter+=1
			
		if entireSet:																#遍历任意alpha值之后没有可以优化的alpha，循环终止
			entireSet=False
		elif alphaPairsChanged==0:
			entireSet=True
	print "iter number:"+str(iter)
	return oS.b,oS.alphas
		
		
if __name__=="__main__":
	dataMat,label=loadData()
	b,alphas=SMO(dataMat,label,0.6,0.001,40)
	print float(b)
		
		
	