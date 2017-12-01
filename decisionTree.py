#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log


"""
	calShannon(dataSet):
	通过香农公式计算给定数据集的熵
	
	splitData(dataSet,index,value):
	根据第index个特征的值是否等于value来划分集合
	
	chooseBestFeature(dataSet)：
	计算每个特征值的的信息熵
	选取具有最大香农熵的特征
	
	majorType(classList):
	如果数据集已经处理了所有属性，但是类标签
	仍旧不是唯一的，采用少数服从多数的方式
	
	createTree(dataSet,labels)：
	根据递归来创建决策树
	
"""

def calShannon(dataSet):
	m=len(dataSet)
	labelDict={}
	for i in dataSet:									#统计各个标签出现的次数
		label=i[-1]										#提取标签
		if label not in labelDict.keys():
			labelDict[label]=1
		else:
			labelDict[label]+=1
	shannon=0.0
	for i in labelDict.keys():
		prob=1.0*labelDict[i]/m
		shannon-=prob*log(prob,2)						#计算香农值
	return shannon
	
def splitData(dataSet,index,value):
	reducedDataSet=[]
	for data in dataSet:
		if data[index]==value:
			reducedVector=data[:index]
			reducedVector.extend(data[index+1:])		#将第index特征等于value的值全部添加到一个新的列表中
			reducedDataSet.append(reducedVector)
	return reducedDataSet
	
	
def chooseBestFeature(dataSet):
	featureNum=len(dataSet[0])-1						#特征值
	baseShannon=calShannon(dataSet)
	#print "baseShannon:"+str(baseShannon)
	bestShannon=-10
	bestFeature=-1
	for i in range(featureNum):
		featureList=[x[i] for x in dataSet]				#第i个特征可以取的特征值
		newShannon=0.0									#代表选择某个特征的熵值
		for feature in featureList:
			subDataSet=splitData(dataSet,i,feature)
			prob=1.0*len(featureList)/len(dataSet)		#取的某个特征的特征值的概率
			newShannon+=prob*calShannon(subDataSet)
			
		#print "newShannon: "+str(newShannon)
		if (baseShannon-newShannon)>bestShannon:
			bestShannon=newShannon
			bestFeature=i
	return bestFeature
	
def majorType(classList):
	classCount={}
	for i in classList:
		if i not in classCount.keys():
			classCount[i]=1
		else:
			classCount[i]+=1 
	sortedList=sorted(classCount.iteritems(),lambda x:x[1],reverse=true)	#按照出现次数的多少进行排序
	return sortedList[0][0]
	
def createTree(dataSet,labels):
	classList=[x[-1] for x in dataSet]										#集合中所有数据的标签集
	if classList.count(classList[0])==len(classList):						#如果完全相同则停止继续划分
		return classList[0]
	if len(dataSet[0])==1:													#如果处理完了所有属性之后，类标签仍旧不是唯一的
		return majorType(classList)											#即数据集中只剩下了类标签，返回出现次数最多的类标签
	bestFeature=chooseBestFeature(dataSet)
	bestFeatureLabel=labels[bestFeature]									#选择最佳分类特征和特征标签
	myTree={bestFeatureLabel:{}}
	del(labels[bestFeature])
	featureValues=[x[bestFeature] for x in dataSet]							#最佳分类特征值的集合
	featureSet=set(featureValues)
	for value in featureSet:
		subLabels=labels[:]
		myTree[bestFeatureLabel][value]=createTree(splitData(dataSet,bestFeature,value),subLabels)
	return myTree
	
if __name__=="__main__":
	dataSet=[[1,1,'y'],[1,1,'y'],[1,0,'n'],[0,1,'n'],[0,1,'n']]
	labels=["A","B"]
	#print str(calShannon(dataSet))
	#print chooseBestFeature(dataSet)
	print str(createTree(dataSet,labels))
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	