#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	stumpClassify(dataSet,dimen,thresh,rule):
	通过阈值比较进行分类，所在阈值一边的数据等于-1，另一边为+1
	根据第dimen个特征进行阈值分类,即单层决策树

	buildStump(dataSet,label,D):
	构建一个单层决策树
	
	
"""


from numpy import *

def loadData():
	dataSet=array([[1.0,2.1],[2.0,1.1],[1.3,1.0],[1.0,1.0],[2.0,1.0]])
	label=[1.0,1.0,-1.0,-1.0,1.0]
	return dataSet,label
	
def stumpClassify(dataSet,dimen,thresh,rule):
	m=shape(dataSet)[0]				#样本数目
	retArrray=ones((m,1))
	if rule=="lt":
		for i in range(m):
			#print dataSet[i][dimen]
			if dataSet[i][dimen]<=thresh:
				retArrray[i][0]=-1
	else:
		for i in range(m):
			if dataSet[i][dimen]>thresh:
				retArrray[i][0]=-1
	return retArrray
	
def buildStump(dataSet,label,D):
	"""
		dataSet:数据集
		label：标签集
		D:样本的权重
	"""
	
	dataSet=dataSet
	#print dataSet[0]
	label=array(label).T
	m,n=shape(dataSet)
	numSteps=10.0																#每个特征由最小值到最大值分类的个数
	bestStump={}																#最佳单层决策树
	bestClass=[]																#依据最佳决策树和说选阈值分类之后的样本标签
	minError=inf																#将最小错误率设置为无穷
	for i in range(n):															#遍历所有特征
		rangeMin=dataSet[:,i].min()
		rangeMax=dataSet[:,i].max()
		stepLength=(rangeMax-rangeMin)/numSteps
		for j in range(-1,int(numSteps)+1):
			for inequal in ["lt","gt"]:
				threshVal=rangeMin+float(j)*stepLength							#每一次根据阈值分类的门限值
				predictedVals=stumpClassify(dataSet,i,threshVal,inequal)		#根据阈值分类之后的预测值
				errorArray=ones((m,1))
				for index in range(m):											#计算predictedArray的预测错误与否
					if label[index]!=predictedVals[index]:
						errorArray[index]=0
				#weightedError=D.T*errorArray									#每个样本的加权错误
				weightedError=1.0*sum(errorArray)/len(errorArray)
				#print weightedError
				if weightedError<minError:										#对于某个特征的某个阈值下的决策树具有更小的错误
					minError=weightedError
					bestStump["dim"]=i											#最佳决策树的分类特征
					bestStump["ineq"]=inequal									#最佳决策树的阈
					bestStump["thresh"]=threshVal								#最佳决策树阈值
					bestClass=predictedVals.copy()								#分类结果
	return bestStump,minError,bestClass
	
def trainDS(dataSet,label,iter):
	"""
		dataset:数据集
		label:类别标签
		iter:迭代次数
	"""
	
	classArr=[]
	m=shape(dataSet)[0]
	D=mat(ones((m,1))/m)														#因为D是一个概率分布向量，所以要除以m
	aggClassEst=mat(zeros((m,1)))
	for i in range(iter):
		bestStump,minError,bestClass=buildStump(dataSet,label,D)
		#alpha=float(0.5*log((1.0-minError)/max(minError,1e-6)))
		alpha=float(0.5*log((1-minError)/minError)
		bestStump["alpha"]=alpha
		classArr.append(bestStump)
		expon=multiply(-1*alpha*mat(label).T,bestClass)
		print alpha
		D=multiply(D,exp(expon))
		D=D/D.sum()
		
		print "\n"
	return classArr
	
if __name__=="__main__":
	dataSet,label=loadData()
	#bestStump,minError,bestClass=buildStump(dataSet,label,mat(ones((5,1)))/5.0)
	#print float(minError)
	trainDS(dataSet,label,40)
	#	print i
	
	
	
		