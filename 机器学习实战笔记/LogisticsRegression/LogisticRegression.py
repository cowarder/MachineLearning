#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	loadData函数:
	将训练文件的数据提取出来
	将每个txt文件中的32*32数据构建为
	1*1024的训练样本
	可以得到一个训练样本矩阵和类别向量
	
	sigmoid函数：
	logistics回归的核心
	
	gradDecent函数：
	计算参数
	
	classify函数：
	根据参数weigh对test样本进行预测
	同时计算错误率
	
	main():
	统一调度函数
"""

from numpy import *
from os import listdir
from sklearn import linear_model

def loadData(directName):
	"""
		directName：数据存储的文件夹名称
	"""
	trainList=listdir(directName)			#获取所有样本的列表
	m=len(trainList)
	dataArray=zeros((m,1024))
	labelArray=zeros((m,1))
	for i in range(m):
		lineData=zeros((1,1024))			#每一个样本的参数
		fileName=trainList[i]				#获取每个样本文件的文件名
		fs=open("%s/%s"%(directName,fileName))
		for j in range(32):
			lineStr=fs.readline()
			for k in range(32):
				lineData[0,32*j+k]=int(lineStr[k])
		dataArray[i,:]=lineData				#第i个样本的参数
		
		fileName=fileName.split(".")[0]		#获取文件名
		label=fileName.split("_")[0]		#获取标签
		
		labelArray[i]=int(label)			#获取第i个样本的标签
	return dataArray,labelArray
	
def sigmoid(x):
	return longfloat(1.0/(1+exp(-x)))					#注意精度
	
def gradDecent(dataArray,labelArray,alpha,maxIteration):
	"""
		dataArray:数据集
		labelArray:结果集
		alpha:步长
		maxIteration:最大迭代次数
	"""
	dataMat=mat(dataArray)		#size:m*n
	labelMat=mat(labelArray)	#size:m*1
	m,n=shape(dataMat)
	weigh=ones((n,1))			#用于存储系数
	for i in range(maxIteration):
		h=sigmoid(dataMat*weigh)	#size:m*1
		error=labelMat-h
		weigh=weigh+alpha*dataMat.T*error
	return weigh
	
def gradientForVector(dataArray,labelArray,alpha,maxIteration):
	"""
		这里的alpha不再是一个实数值，而是一个向量(10*m)
	"""
	dataMat=mat(dataArray)
	m,n=shape(dataMat)
	weigh=ones((10,n))
	for i in range(maxIteration):
		h=sigmoid(dataMat*weigh.T)		#m*10
		error=labelMat-h				#m*10
		weigh=weigh+alpha*dataMat.T*error
		
	
def classify(testdir,weigh):
	"""
		testdir:测试数据的文件夹
		weigh：给定的参数向量
	"""
	dataArray,labelArray=loadData(testdir)
	dataMat=mat(dataArray)
	labelMat=mat(labelArray)
	testResults=sigmoid(dataMat*weigh)		#size:m*1
	m=len(testResults)
	error=0.0
	for i in range(m):
		if testResults[i]>0.5:
			#print str(labelMat[i])+" is test as "+str(testResults[i])+"\n"
			if labelMat[i]!=1:
				error+=1
		
		else:
			#print str(labelMat[i])+" is test as "+str(testResults[i])+"\n"
			if labelMat[i]!=0:
				error+=1
		
	print "error rate:"+str(error/m)
	
def main(trainDir,testDir,alpha=0.7,maxIteration=10):
	"""
		trainDir:训练集
		testDir:测试集
		alpha：步长
		maxIteration:最大迭代次数
	"""
	dataArray,labelArray=loadData(trainDir)
	weigh=gradDecent(dataArray,labelArray,alpha,maxIteration)
	classify(testDir,weigh)
	
	#cla=linear_model.Ridge(alpha=0.1)
	#cla.fit(dataArray,labelArray)
	#print cla.coef_
	
	
if __name__=="__main__":
	main("train","test",0.7,10)
	
	
	
	