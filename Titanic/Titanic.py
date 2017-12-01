#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据格式：
 
乘客id		是否生还   乘客类型  名字	性别	年龄	兄弟加配偶数目		父母孩子数目	 票号		票价	船舱号		上船港口号
PassengerId	Survived	Pclass	 Name	Sex		Age			SibSp				Parch		Ticket		Fare	Cabin		Embarked

"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import re

def toInt(l):
	for i in range(len(l)):
		l[i]=int(l[i])
	return l

def toFloat(l):
	for i in range(len(l)):
		if len(l[i])!=0:
			l[i]=float(l[i])
		else:
			l[i]=-1.0
	return l

def saveOneDec(num):
	return float('%.2f'%num)
	
def linePlot(x,y):
	plt.plot(x,y)
	plt.show()
	
def doubleBar(n_groups,x,y,fName,sName,title):
	#获得一个图像fig，一个1*1的子窗口
	fig, ax = plt.subplots(1,1)
	#等差数组
	index = np.arange(n_groups)
	bar_width = 0.35
	rects1 = plt.bar(index, x, bar_width,alpha=0.4, color='b',label=fName)
	rects2 = plt.bar(index+ bar_width, y, bar_width,alpha=0.4,color='r',label=sName) 
	#在x轴上面对每个bar进行说明
	plt.xticks(index+bar_width, map(str,index))  
	#plt.ylim(0,500)
	plt.legend()
	plt.tight_layout()
	plt.title(title)
	plt.show()
	
def singleBar(n_groups,xNames,x,title):
	fig, ax = plt.subplots(1,1)
	#等差数组
	index = np.arange(n_groups)
	bar_width = 0.35
	plt.bar(index, x, bar_width,alpha=0.4, color='b')
	#在x轴上面对每个bar进行说明
	plt.xticks(index+bar_width, xNames)  
	#plt.ylim(0,500)
	plt.legend()
	plt.tight_layout()
	plt.title(title)
	plt.show()
	
def loadData():
	Num=[]
	Label=[]
	Pclass=[]
	Name=[]
	Sex=[]
	Age=[]
	SibSp=[]
	Parch=[]
	Ticket=[]
	Fare=[]
	Cabin=[]
	Embarked=[]
	with open("train.csv") as f:
		lines=csv.reader(f)
		for line in lines:
			Num.append(line[0])
			Label.append(line[1])
			Pclass.append(line[2])
			Name.append(line[3])
			Sex.append(line[4])
			Age.append(line[5])
			SibSp.append(line[6])
			Parch.append(line[7])
			Ticket.append(line[8])
			Fare.append(line[9])
			Cabin.append(line[10])
			Embarked.append(line[11])
	del(Num[0])
	del(Label[0])
	del(Pclass[0])
	del(Name[0])
	del(Sex[0])
	del(Age[0])
	del(SibSp[0])
	del(Parch[0])
	del(Ticket[0])
	del(Fare[0])
	del(Cabin[0])
	del(Embarked[0])
	
	return Num,toInt(Label),toInt(Pclass),Name,Sex,toFloat(Age),toInt(SibSp),toInt(Parch),Ticket,toFloat(Fare),Cabin,Embarked
	
def loadTestData():
	Num=[]
	Pclass=[]
	Name=[]
	Sex=[]
	Age=[]
	SibSp=[]
	Parch=[]
	Ticket=[]
	Fare=[]
	Cabin=[]
	Embarked=[]
	with open("test.csv") as f:
		lines=csv.reader(f)
		for line in lines:
			Num.append(line[0])
			Pclass.append(line[1])
			Name.append(line[2])
			Sex.append(line[3])
			Age.append(line[4])
			SibSp.append(line[5])
			Parch.append(line[6])
			Ticket.append(line[7])
			Fare.append(line[8])
			Cabin.append(line[9])
			Embarked.append(line[10])
	del(Num[0])
	del(Pclass[0])
	del(Name[0])
	del(Sex[0])
	del(Age[0])
	del(SibSp[0])
	del(Parch[0])
	del(Ticket[0])
	del(Fare[0])
	del(Cabin[0])
	del(Embarked[0])
	
	return toInt(Num),toInt(Pclass),Name,Sex,toFloat(Age),toInt(SibSp),toInt(Parch),Ticket,toFloat(Fare),Cabin,Embarked


def analyseName(nameList):
	"""
		分析乘客名字中的潜在关系：一个家庭的人会不会同时死亡
		
	"""
	
	splitName=[]
	for name in nameList:
		mode=re.compile(r"\s")
		splitedName=mode.split(name)
		#print splitedName
		splitName.append(splitedName[0])
	#这里分析的是同一个家族姓氏的个数
	s=set(splitName)
	#print len(s)

def analyseFamilySize(Label,SibSp,Parch):
	"""
		这里分析的是家庭成员的所少与死亡与否的关系
	"""	
	familySize=range(0,12)
	dieNum=np.zeros(12)
	surviveNum=np.zeros(12)
	for i in range(len(Label)):
		if Label[i]==0:
			dieNum[SibSp[i]+Parch[i]]+=1
		elif Label[i]==1:
			surviveNum[SibSp[i]+Parch[i]]+=1
	
	doubleBar(12,dieNum,surviveNum,'dieNum','surviveNum',"familySize-surviveNum")
	
	
	#打印一个柱状图的
	#fig=plt.figure()
	#plt.bar(familySize,dieNum,0.4,color="red")
	#plt.xlabel("familySize")
	#plt.ylabel("dieNum")
	#plt.title("familySize-survive")
	#plt.show()
	
def analyseSex(Label,Sex):
	"""
		性别与生还关系
	"""
	mSurvive=0
	mDie=0
	fSurvive=0
	fDie=0
	for i in range(len(Label)):
		if Label[i]==1:
			if Sex[i]=='male':
				mSurvive+=1
			elif Sex[i]=='female':
				fSurvive+=1
		elif Label[i]==0:
			if Sex[i]=='male':
				mDie+=1
			elif Sex[i]=='female':
				fDie+=1
	doubleBar(2,[mDie,mSurvive],[fDie,fSurvive],'male','female','sex-survive')
	
def analyseAgeWithSurvival(Label,Age):
	"""
		分析不同年龄段的生还关系
	"""
	D1=0
	D2=0
	D3=0
	D4=0
	D5=0
	D6=0
	D7=0
	D8=0
	S1=0
	S2=0
	S3=0
	S4=0
	S5=0
	S6=0
	S7=0
	S8=0
	for i in range(len(Label)):
		if Label[i]==1:
			if Age[i]>0 and Age[i]<10:
				S1+=1
			elif Age[i]<20:
				S2+=1
			elif Age[i]<30:
				S3+=1
			elif Age[i]<40:
				S4+=1
			elif Age[i]<50:
				S5+=1
			elif Age[i]<60:
				S6+=1
			elif Age[i]<70:
				S7+=1
			else:
				S8+=1
		elif Label[i]==0:
			if Age[i]>0 and Age[i]<10:
				D1+=1
			elif Age[i]<20:
				D2+=1
			elif Age[i]<30:
				D3+=1
			elif Age[i]<40:
				D4+=1
			elif Age[i]<50:
				D5+=1
			elif Age[i]<60:
				D6+=1
			elif Age[i]<70:
				D7+=1
			else:
				D8+=1
	#print D5,D7
	doubleBar(8,[D1,D2,D3,D4,D5,D6,D7,D8],[S1,S2,S3,S4,S5,S6,S7,S8],'die','survive','age-survival')

def analyseAgeWithSex(Sex,Age):
	"""
		不同性别的平均年龄,实验结果发现男女差别大约为3
	"""
	maleAge=[]
	femaleAge=[]
	for i in range(len(Sex)):
		if Age[i]>=0:
			if Sex[i]=='male':
				maleAge.append(Age[i])
			elif Sex[i]=='female':
				femaleAge.append(Age[i])
	print "maleAverAge:"+str(sum(maleAge)/len(maleAge))
	print "femaleAverAge:"+str(sum(femaleAge)/len(femaleAge))

def analyseAgeByName(Name,Age):
	MrNum=[]
	MissNum=[]
	MrsNum=[]
	for i in range(len(Name)):
		if Age[i]>=0:
			if 'Mr.' in Name[i]:
				MrNum.append(Age[i])
			elif 'Mrs.' in Name[i]:
				MrsNum.append(Age[i])
			elif 'Miss.' in Name[i]:
				MissNum.append(Age[i])
	return 1.0*sum(MrNum)/len(MrNum),1.0*sum(MrsNum)/len(MrsNum),1.0*sum(MissNum)/len(MissNum)
	
	
def analyseAgeWithPclass(Pclass,Age):
	"""
		发现在不同等级的乘客年龄存在差距
	"""
	class1=[]
	class2=[]
	class3=[]
	for i in range(len(Sex)):
		if Age[i]>=0:
			if Pclass[i]==1:
				class1.append(Age[i])
			elif Pclass[i]==2:
				class2.append(Age[i])
			elif Pclass[i]==3:
				class3.append(Age[i])
	print sum(class1)/len(class1)
	print sum(class2)/len(class2) 
	print sum(class3)/len(class3)
	
def getAgeDiv(Sex,Pclass,Age):
	"""
		根据已有的乘客等级和乘客的性别求出年龄的平均值，填充缺失年龄的乘客
	"""
	m1=[]
	m2=[]
	m3=[]
	f1=[]
	f2=[]
	f3=[]
	for i in range(len(Sex)):
		if Age[i]!='':
			if Sex[i]=='male':
				if Pclass[i]==1:
					m1.append(Age[i])
				elif Pclass[i]==2:
					m2.append(Age[i])
				elif Pclass[i]==3: 
					m3.append(Age[i])
			elif Sex[i]=='female':
				if Pclass[i]==1:
					f1.append(Age[i])
				elif Pclass[i]==2:
					f2.append(Age[i])
				elif Pclass[i]==3:
					f3.append(Age[i])
	m1Aver=saveOneDec(sum(m1)/len(m1))
	m2Aver=saveOneDec(sum(m2)/len(m2))
	m3Aver=saveOneDec(sum(m3)/len(m3))
	f1Aver=saveOneDec(sum(f1)/len(f1))
	f2Aver=saveOneDec(sum(f2)/len(f2))
	f3Aver=saveOneDec(sum(f3)/len(f3))
	#print m1Aver,m2Aver,m3Aver,f1Aver,f2Aver,f3Aver
	return m1Aver,m2Aver,m3Aver,f1Aver,f2Aver,f3Aver

	
def getMeans(Pclass,Fare,Embarked):
	"""
		获取不同乘客类型在不同港口上船的各平均费用
	"""
	
	fCList=[]
	fCMean=0.0
	fSList=[]
	fSMean=0.0
	fQList=[]
	fQMean=0.0
	sCList=[]
	sCMean=0.0
	sSList=[]
	sSMean=0.0
	sQList=[]
	sQMean=0.0
	tCList=[]
	tCMean=0.0
	tSList=[]
	tSMean=0.0
	tQList=[]
	tQMean=0.0
	for i in range(len(Pclass)):
		if Fare[i]>=0 and Embarked[i]!="":
			if Pclass[i]==1:
				if Embarked[i]=='C':
					fCList.append(Fare[i])
				elif Embarked[i]=='S':
					fSList.append(Fare[i])
				elif Embarked[i]=='Q':
					fQList.append(Fare[i])
			elif Pclass[i]==2:
				if Embarked[i]=='C':
					sCList.append(Fare[i])
				elif Embarked[i]=='S':
					sSList.append(Fare[i])
				elif Embarked[i]=='Q':
					sQList.append(Fare[i])
			elif Pclass[i]==3:
				if Embarked[i]=='C':
					tCList.append(Fare[i])
				elif Embarked[i]=='S':
					tSList.append(Fare[i])
				elif Embarked[i]=='Q':
					tQList.append(Fare[i])
	fCMean=1.0*sum(fCList)/len(fCList)
	fSMean=1.0*sum(fSList)/len(fSList)
	fQMean=1.0*sum(fQList)/len(fQList)
	sCMean=1.0*sum(sCList)/len(sCList)
	sSMean=1.0*sum(sSList)/len(sSList)
	sQMean=1.0*sum(sQList)/len(sQList)
	tCMean=1.0*sum(tCList)/len(tCList)
	tSMean=1.0*sum(tSList)/len(tSList)
	tQMean=1.0*sum(tQList)/len(tQList)
	
	"""
	cList=fCList
	cList.extend(sCList)
	cList.extend(tCList)
	cList=sorted(cList)
	x=np.linspace(0,max(cList),len(cList))
	linePlot(x,cList)
	"""
	
	return fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean
	
	
def preEmbarkByFare(pclass,fare,fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean):
	"""
		通过缴纳的费用预测上船时的港口
	"""
	
	if pclass==1:
		mean = [x for x in [fCMean,fSMean,fQMean] if abs(x-fare)==min(abs(fare-fCMean),abs(fare-fSMean),abs(fare-fQMean))][0]
	elif pclass==2:
		mean = [x for x in [sCMean,sSMean,sQMean] if abs(x-fare)==min(abs(fare-sCMean),abs(fare-sSMean),abs(fare-sQMean))][0]
	elif pclass==3:
		mean = [x for x in [tCMean,tSMean,tQMean] if abs(x-fare)==min(abs(fare-tCMean),abs(fare-tSMean),abs(fare-tQMean))][0]
	#print mean
	if mean in [fCMean,sCMean,tCMean]:
		return 'C'
	elif mean in [fSMean,sSMean,tSMean]:
		return 'S'
	elif mean in [fQMean,sQMean,tQMean]:
		return 'Q'

def preFareByEmbark(pclass,embark,fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean):
	if pclass==1:
		if embark=='S':
			return fSMean
		elif embark=='C':
			return fCMean
		elif embark=='Q':
			return fQMean
	elif pclass==2:
		if embark=='S':
			return sSMean
		elif embark=='C':
			return sCMean
		elif embark=='Q':
			return sQMean
	elif pclass==3:
		if embark=='S':
			return tSMean
		elif embark=='C':
			return tCMean
		elif embark=='Q':
			return tQMean
			
			
def completeAge(Age,Pclass,Sex):
	"""
		填充缺失年龄的乘客，返回新的列表
	"""
	
	m1Aver,m2Aver,m3Aver,f1Aver,f2Aver,f3Aver=getAgeDiv(Sex,Pclass,Age)
	
	for i in range(len(Age)):
		if Age[i]<=0.0:
			if Sex[i]=='male':
				if Pclass[i]==1:
					Age[i]=m1Aver
				elif Pclass[i]==2:
					Age[i]=m2Aver
				elif Pclass[i]==3:
					Age[i]=m3Aver
			elif Sex[i]=='female':
				if Pclass[i]==1:
					Age[i]=f1Aver
				elif Pclass[i]==2:
					Age[i]=f2Aver
				elif Pclass[i]==3:
					Age[i]=f3Aver
	return Age

def completeAge1(Age,Name):
	"""
		根据乘客的名字预测乘客的年龄
	"""
	MrAverAge,MrsAverAge,MissAverAge=analyseAgeByName(Name,Age)
	
	for i in range(len(Age)):
		if Age[i]<0:
			if 'Mr.' in Name[i]:
				Age[i]=MrAverAge
			elif 'Mrs.' in Name[i]:
				Age[i]=MrsAverAge
			elif 'Miss.' in Name[i]:
				Age[i]=MissAverAge
	return Age	
	
def completeFare(Fare,Embarked,fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean):
	"""
		填充缺失的乘客费用
	"""
	for i in range(len(Fare)):
		if Fare[i]<0:
			Fare[i]=preFareByEmbark(Pclass[i],Embarked[i],fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean)
	return Fare
	
def completeEmbark(Pclass,Fare,Embarked,fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean):
	"""
		填充乘客缺失的上船信息
	"""
	for i in range(len(Embarked)):
		if Embarked[i]=='':
			Embarked[i]=preEmbarkByFare(Pclass[i],Fare[i],fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean)
	return Embarked
	
def facterize(Num,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked):
	"""
		将特征因子化
	"""
	
	maxFare=max(Fare)
	minFare=min(Fare)
	maxAge=max(Age)
	minAge=min(Age)
	p1=[]
	p2=[]
	p3=[]
	sexMale=[]
	sexFemale=[]
	cabinExist=[]
	embarkS=[]
	embarkC=[]
	embarkQ=[]
	for i in range(len(Num)):
		if Pclass[i]==1:
			p1.append(1)
			p2.append(0)
			p3.append(0)
		elif Pclass[i]==2:
			p1.append(0)
			p2.append(1)
			p3.append(0)
		elif Pclass[i]==3:
			p1.append(0)
			p2.append(0)
			p3.append(1)
			
		#特征缩放
		Fare[i]=(float(Fare[i]-minFare)/(maxFare-minFare))
		Age[i]=(float(Age[i]-minAge)/(maxAge-minAge))
		
		if Sex[i]=='male':
			sexMale.append(1)
			sexFemale.append(0)
		elif Sex[i]=='female':
			sexMale.append(0)
			sexFemale.append(1)
			
		if Cabin[i]!='':
			cabinExist.append(1)
		elif Cabin[i]=='':
			cabinExist.append(0)
		
		if Embarked[i]=='S':
			embarkC.append(0)
			embarkQ.append(0)
			embarkS.append(1)
		elif Embarked[i]=='C':
			embarkC.append(1)
			embarkQ.append(0)
			embarkS.append(0)
		elif Embarked[i]=='Q':
			embarkC.append(0)
			embarkQ.append(1)
			embarkS.append(0)
	return Num,p1,p2,p3,Name,sexMale,sexFemale,Age,SibSp,Parch,Ticket,Fare,cabinExist,embarkC,embarkQ,embarkS
			
def getTrainFormData(Label,p1,p2,p3,sexMale,sexFemale,Age,SibSp,Parch,Fare,cabinExist,embarkC,embarkQ,embarkS):
	"""
		将训练的数据形式化
	"""
	label=np.zeros((len(Label),1))
	dataSet=np.zeros((len(Label),13))
	for i in range(len(Label)):
		label[i][0]=Label[i]
		dataSet[i][0]=p1[i]
		dataSet[i][1]=p2[i]
		dataSet[i][2]=p3[i]
		dataSet[i][3]=sexMale[i]
		dataSet[i][4]=sexFemale[i]
		dataSet[i][5]=Age[i]
		dataSet[i][6]=SibSp[i]
		dataSet[i][7]=Parch[i]
		dataSet[i][8]=Fare[i]
		dataSet[i][9]=cabinExist[i]
		dataSet[i][10]=embarkC[i]
		dataSet[i][11]=embarkQ[i]
		dataSet[i][12]=embarkS[i]
	return dataSet,label
	
def getHandleFormData(Num,p1,p2,p3,Name,sexMale,sexFemale,Age,SibSp,Parch,Ticket,Fare,cabinExist,embarkC,embarkQ,embarkS):
	"""
		将要预测的数据形式化
	"""
	
	dataSet=np.zeros((len(Num),13))
	for i in range(len(Num)):
		dataSet[i][0]=p1[i]
		dataSet[i][1]=p2[i]
		dataSet[i][2]=p3[i]
		dataSet[i][3]=sexMale[i]
		dataSet[i][4]=sexFemale[i]
		dataSet[i][5]=Age[i]
		dataSet[i][6]=SibSp[i]
		dataSet[i][7]=Parch[i]
		dataSet[i][8]=Fare[i]
		dataSet[i][9]=cabinExist[i]
		dataSet[i][10]=embarkC[i]
		dataSet[i][11]=embarkQ[i]
		dataSet[i][12]=embarkS[i]
	return dataSet,Num
	

	
def sigmoid(x):
	return np.longfloat(1.0/(1+np.exp(-x)))
	
def gradDecent(dataArray,labelArray,alpha,maxIteration):
	"""
		dataArray:数据集
		labelArray:结果集
		alpha:步长
		maxIteration:最大迭代次数
	"""
	dataMat=np.mat(dataArray)		#size:m*n
	labelMat=np.mat(labelArray)	#size:m*1
	m,n=np.shape(dataMat)
	weigh=np.ones((n,1))			#用于存储系数
	for i in range(maxIteration):
		h=sigmoid(dataMat*weigh)	#size:m*1
		error=labelMat-h
		weigh=weigh+alpha*dataMat.transpose()*error
	return weigh
	
def train(dataSet,label):
	return gradDecent(dataSet,label,0.002,800)

def test(weight,dataSet,label):
	result=sigmoid(dataSet*weigh)
	error=0.0
	for i in range(len(result)):
		if result[i]>0.5:
			if label[i]!=1:
				error+=1
		else:
			if label[i]!=0:
				error+=1
	#print error/len(label)
	return error/len(label)
	
def getPreResult(Num,dataSet,weigh):
	"""
		获取对test文件中样本的预测值
	"""
	pre=[]
	result=sigmoid(dataSet*weigh)
	for i in range(len(dataSet)):
		if result[i]>0.5:
			pre.append(1)
		else:
			pre.append(0)
	return Num,pre
	
def saveData(Num,pre):
	newData=[]
	for i in range(len(Num)):
		newData.append([Num[i],str(pre[i])])
	with open("result.csv",'wb') as f:
		writer=csv.writer(f)
		writer.writerow(['PassengerId','Survived'])
		for i in range(len(Num)):
			writer.writerow(newData[i])
			
	
if __name__=="__main__":
	Num,Label,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked=loadData()
	fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean=getMeans(Pclass,Fare,Embarked)
	
	
	analyseAgeByName(Name,Age)
	#Age=completeAge(Age,Pclass,Sex)
	Age=completeAge1(Age,Name)
	Embarked=completeEmbark(Pclass,Fare,Embarked,fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean)
	Fare=completeFare(Fare,Embarked,fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean)
	
	Num,p1,p2,p3,Name,sexMale,sexFemale,Age,SibSp,Parch,Ticket,Fare,cabinExist,embarkC,embarkQ,embarkS=facterize(Num,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked)
	dataSet,label=getTrainFormData(Label,p1,p2,p3,sexMale,sexFemale,Age,SibSp,Parch,Fare,cabinExist,embarkC,embarkQ,embarkS)
	weigh=train(dataSet,label)

	averError=0.0
	#十折交叉验证
	startIndex=0
	for i in range(10):
		trainSet=[]
		for j in range(startIndex,startIndex+89):
			trainSet.append(dataSet[j])
		for k in range(startIndex+89,891):
			trainSet.append(dataSet[i])
			
		testSet=dataSet[startIndex:startIndex+89]
		testLabel=label[startIndex:startIndex+89]
		trainLabel=[]
		for j in range(startIndex,startIndex+89):
			trainLabel.append(label[j])
		for k in range(startIndex+89,891):
			trainLabel.append(label[k])
		startIndex+=89
		weigh=train(trainSet,trainLabel)
		error=test(weigh,testSet,testLabel)
		averError+=error
	averError/=10
	
	print averError
	#analyseName(Name)
	#analyseFamilySize(Label,SibSp,Parch)
	#analyseSex(Label,Sex)	
	#analyseAgeWithSurvival(Label,Age)
	#print fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean
	#print preEmbarkByFare(3,7.7958,fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean)
	#print len([x for x in ])
	#analyseAgeWithSex(Sex,Age)
	#analyseAgeWithPclass(Pclass,Age)
	#print Embarked
	"""
	print len(Num)
	print len(Label)
	print len(p1)
	print len(p2)
	print len(p3)
	print len(Name)
	print len(sexMale)
	print len(sexFemale)
	print len(Age)
	print len(embarkC)
	print len(embarkQ)
	print len(embarkS)
	print len(Parch)
	print len(Ticket)
	print len(cabinExist)
	print Fare
	"""
	
	Num,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked=loadTestData()
	fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean=getMeans(Pclass,Fare,Embarked)
	Age=completeAge(Age,Pclass,Sex)
	Embarked=completeEmbark(Pclass,Fare,Embarked,fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean)
	Fare=completeFare(Fare,Embarked,fCMean,fSMean,fQMean,sCMean,sSMean,sQMean,tCMean,tSMean,tQMean)
	#print len(Age),len(Fare),len(Embarked)
	Num,p1,p2,p3,Name,sexMale,sexFemale,Age,SibSp,Parch,Ticket,Fare,cabinExist,embarkC,embarkQ,embarkS=facterize(Num,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked)
	dataSet,Num=getHandleFormData(Num,p1,p2,p3,Name,sexMale,sexFemale,Age,SibSp,Parch,Ticket,Fare,cabinExist,embarkC,embarkQ,embarkS)
	Num,pre=getPreResult(Num,dataSet,weigh)
	saveData(Num,pre)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
		
		
		