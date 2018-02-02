#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	knn算法
	
	classify0:
	k邻近算法，计算两个向量点之间的距离，根据距离的大小进行排序
	选取k个距离最小的点，查看在这k个点中标签出现频率最高的点
	
	inX:待测定的样本向量
	dataSet:训练样本
	label:标签
	k:k邻近算法的参数k
	
	fileToMatrix:
	filename：提取数据的文件名
	将文件中的数据转换为矩阵的形式
	
	autoNorm：
	归一化数据，(oldvalue-min)/(max-min)
	将数据统一归一到特定的范围内
"""
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataset():
	group=array([[1.0,1.1],[1.0,1.1],[0,0],[0,0.1]])
	label=['A','A','B','B']
	return group,label

def classify0(intX,dataSet,label,k):
	#dataSet=autoNorm(dataSet)
	xArray=tile(intX,(dataSet.shape[0],1))
	subArray=xArray-dataSet
	squareArray=subArray**2
	disArray=(squareArray.sum(1))**0.5
	#for i in disArray:
	#	print i
	classCount={}
	s=sorted(zip(disArray,label),key=lambda x:x[0])		#根据距离的大小排序data和label
	#print s
	for i in range(k):
		if classCount.get(s[i][1],0)==0:
			classCount[s[i][1]]=1
		else:
			classCount[s[i][1]]+=1
	print classCount
	for key,value in classCount.items():
		print key,value
		print "\n"

	predict=max(classCount.items(),key=lambda x:x[1])[0]
	
	return predict
	
	
def fileToMatrix(filename):
	fs=open(filename)
	lines=fs.readlines()
	m=len(lines)
	dataMatrix=zeros((m,3))		#这里的3指的是样本参数个数
	labels=[]
	for i in range(m):
		stripLine=lines[i].strip()
		listLine=stripLine.split("\t")		#提取参数
		dataMatrix[i,:]=listLine[0:3]
		label=listLine[-1]		#获取标签
		labels.append(label)
	return dataMatrix,labels
		
		
def autoNorm(dataSet):
	maxNums=dataSet.max(0)		#取每一列的最大值
	minNums=dataSet.min(0)		#取每一列的最小值
	rangeNums=maxNums-minNums	#每一列的范围值
	normSet=zeros(dataSet.shape)
	normSet=dataSet-tile(minNums,(dataSet.shape[0],1))		#减去最小值
	normSet=normSet/tile(rangeNums,(dataSet.shape[0],1))
	return normSet
	
def testData():
	data,label=fileToMatrix("datingTestSet.txt")
	error=0.0
	for i in range(0,999):
		if classify0(data[i],data,label,1000)!=label[i]:
			error+=1
	print "error rate:"+str(error/1000)
	
if __name__=="__main__":
	#group,label=createDataset()
	#classify0([0,0],group,label,3)
	testData()
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	