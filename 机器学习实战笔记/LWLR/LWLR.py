import numpy as np
import matplotlib.pyplot as plt

def loadData(fileName):
	numFeat=len(open(fileName).readline().split('\t'))
	dataMat=[]
	label=[]
	f=open(fileName)
	for line in f.readlines():
		lineArr=line.strip().split('\t')
		dataMat.append([float(x) for x in lineArr[:-1]])
		label.append(float(lineArr[-1]))
	return dataMat,label
	
def standRegress(dataMat,label):
	"""
		利用公式w=(X.T*X)*X.T*y计算变量的权值
	"""
	xTx=dataMat.T*dataMat
	if np.linalg.det(xTx)==0:
		print "This matrix cant not be reverse."
		return
	w=xTx.I*(dataMat.T*label)
	return w

def lwlr(testData,dataMat,label,k):
	"""
		加权线性回归
	"""
	m=np.shape(dataMat)[0]
	weights=np.mat(np.eye(m))
	for i in range(m):
		differMat=testData-dataMat[i]
		weights[i,i]=np.exp((differMat*differMat.T)/(-2.0*k**2))
	xTx=dataMat.T*weights*dataMat
	if np.linalg.det(xTx)==0:
		print "This matrix cant not be reverse."
		return
	w=xTx.I*dataMat.T*weights*label
	#返回的是预测值
	return testData*w
	
def getArrWs(testData,dataMat,label,k):
	m=np.shape(testData)[0]
	result=np.zeros(m)
	for i in range(m):
		result[i]=lwlr(testData[i],dataMat,label,k)
	return result
	
def calError(preLabel,label):
	return ((preLabel-label)**2).sum()
	
def ridgeRegression(dataMat,label,lam=0.2):
	"""
		岭回归:应用于特征多于变量的情况
	"""
	xTx=dataMat.T*dataMat
	demon=xTx+lam*np.eye(dataMat.shape[1])
	if np.linalg.det(demon)==0:
		print "This matrix cant not be reverse."
		return
	w=demon.I*dataMat.T*label
	return w


if __name__=="__main__":
	dataMat,label=loadData("ex0.txt")
	dataMat=np.mat(dataMat)
	label=np.mat(label).T
	w=standRegress(dataMat,label)
	#w=lwlr(dataMat[0],dataMat,label,1)
	#print np.corrcoef(preDataMat,preDataMat)
	#for i in range(len(dataMat)):
	#	print (dataMat[i]*w)[0,0],label[i,0]
	preLabel=getArrWs(dataMat,dataMat,label,0.01)
	
	w1=ridgeRegression(dataMat,label)
	print w1
	"""
	fig=plt.figure()
	pic=fig.add_subplot(1,1,1)
	x=[i[0,0] for i in dataMat[:,1]]
	y=[j[0,0] for j in label[:,0]]
	pic.scatter(x,y)
	data=[(a,b) for a,b in zip(x,preLabel)]#首先转换为元组
	data.sort()
	x=[a for a,b in data]
	y=[b for a,b in data]
	for i in range(len(x)):
		print y[i],label[i]
	pic.plot(x,y)
	plt.show()
	"""