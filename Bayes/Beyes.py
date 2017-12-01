#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
	getDivWords:
	遍历所有的邮件，提取出所有的不重复单词
	
	calSetTime:
	计算每个data数据中目标单词是否出现将邮件转换
	为标准向量
	
	calBagTime：
	根据设定好的提取向量，将邮件转换为的标准向量
	
	train:
	训练数据，根据贝叶斯公式返回邮件为垃圾邮件的
	概率和分别在垃圾邮件和非垃圾邮件的情况下各个
	单词出现的概率
	
	test：
	测试函数
	
	textParse:
	解析文本，返回文本中根据正则规范且长度大于3
	分割出的字符串
	
	spamTest：
	垃圾邮件判断函数
"""

from numpy import *
import re
import random

def loadData():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
	['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
	['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
	['stop', 'posting', 'stupid', 'worthless', 'garbage'],
	['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
	['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	
	classVec = [0,1,0,1,0,1]
	return postingList,classVec


def getDivWords(dataSet):
	words=set([])
	for i in dataSet:
		words=words|set(i)					#取并集
	return list(words)
	
#set if word model
def calSetTime(wordList,data):
	result=zeros(len(wordList))
	for word in data:
		if word in wordList:
			result[wordList.index(word)]=1	#统计单词是否在邮件中出现
	return result
	
#bag of word model
def calBagTime(wordList,data):
	result=zeros(len(wordList))
	for word in data:
		if word in wordList:
			result[wordList.index(word)]+=1	#统计单词在一封邮件中出现的次数
	return result
	
def train(dataSet,label):
	m=len(dataSet)
	n=len(dataSet[0])
	posProb=1.0*sum(label)/len(label)		#结果为1的概率
	
	posNum=2.0
	negNum=2.0								#标签属于同一类的邮件中所有关键词出现的总次数
	posVec=ones(n)							#统计所有关键字在多少邮件中出现
	negVec=ones(n)							#这里不取总次数和单个单词数目为0是为了在
											#计算文档属于某个类别的概率时不因为某个单词没有出现导致概率为0
	
	for i in range(m):
		if label[i]==1:
			posVec+=dataSet[i]
			#posNum+=sum(dataSet[i])
			posNum+=1
		else:
			#negNum+=sum(dataSet[i])
			negNum+=1
			negVec+=dataSet[i]
			
	prob_1=posVec/posNum					#单个关键词在标签为1的邮件中出现的邮件数目/所有标签为1的邮件中出现的关键词总数
	prob_0=negVec/negNum					#即在给定文档类别条件下词汇表中各个单词出现的概率
	return posProb,log(prob_1),log(prob_0)	#这里使用log是为了避免因为概率太小而出现相乘结果下溢的情况
	
def classify(vect,posProb,prob_1,prob_0):
	p1=sum(vect*prob_1)+log(posProb)		#分别计算在vect变量出现的条件下结果为1或0的概率
	p0=sum(vect*prob_0)+log(posProb)
	if p1>p0:
		return 1
	else:
		return 0
	
	
def test():
	dataSet,label=loadData()
	divWords=getDivWords(dataSet)			#获取所有不重复单词
	changedSet=[]							#用于存放修改后的数据集
	for i in dataSet:
		wordList=calBagTime(divWords,i)		#将每一个样本转换为数字的格式
		changedSet.append(wordList)			#构成一个参数个数为不重复单词数的新的数组
	posProb,prob_1,prob_0=train(changedSet,label)
	vect=["love","my","dalmation"]
	paraVector=calBagTime(divWords,vect)		#转换为包含所有关键字的向量
	classify(paraVector,posProb,prob_1,prob_0)
	vect1=["stupid","garbage"]
	paraVector1=calBagTime(divWords,vect1)
	classify(paraVector1,posProb,prob_1,prob_0)
	
def textParse(text):
	mode=re.compile(r"/W*")
	words=mode.split(text)
	return [word.lower() for word in words if len(word)>3 ]
	
def spamTest():
	emailList=[]
	labelList=[]
	words=[]
	
	for i in range(1,26):
		wordList=textParse(open("C:/Users/ASUS/Desktop/data/spam/%d.txt"%i).read())		#读取每个垃圾文件的文本内容并解析
		emailList.append(wordList)
		labelList.append(1)
		words.extend(wordList)															#不会将数组作为元素添加进去，而是将数组中的元素拆分后添加
		wordList=textParse(open("C:/Users/ASUS/Desktop/data/ham/%d.txt"%i).read())		#读取每个非垃圾文件的文本内容并解析
		emailList.append(wordList)
		labelList.append(0)
		words.extend(wordList)
	divWords=getDivWords(emailList)														#获取在所有邮件中不重复出现的单词
	trainIndex=range(50)
	testIndex=[]																		#用于存储测试样本下标
	
	for i in range(10):																	#随机选取10个作为测试样本
		j=int(random.uniform(0,len(trainIndex)))										#随机选取测试邮件的下标
		testIndex.append(trainIndex[j])
		del(trainIndex[j])																#从训练集中删去
	
	trainMat=[]
	labelMat=[]
	for i in range(len(trainIndex)):
		j=trainIndex[i]
		trainMat.append(calBagTime(divWords,emailList[j]))
		labelMat.append(labelList[j])
	posProb,prob_1,prob_0=train(trainMat,labelMat)
	error=0.0
	for i in range(len(testIndex)):
		j=testIndex[i]
		if classify(calBagTime(divWords,emailList[j]),posProb,prob_1,prob_0)==1:
			if labelList[j]!=1:
				error+=1
		else:
			if labelList[j]!=0:
				error+=1
	print "error rate:"+str(error/len(labelList))

		
if __name__=="__main__":
	spamTest()
	
	
	
	
	
	
	
	
	
	