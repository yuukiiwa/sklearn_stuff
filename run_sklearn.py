import sys
import numpy as np
fn=sys.argv[1]

def genEntry(L):
 entry=[]
 for i in L:
  entry.append(int(i))
 return entry

def openFile(fn):
 file=open(fn,"r")
 file.readline()
 dict={'data':[],'target':[]}
 for ln in file:
  ln=ln.strip("\r\n").split(",")
#  dict['target'].append(int(round(float(ln[0]),1)*10))  #classifier input
  dict['target'].append(float(ln[0]))  #regressor input
  entry=genEntry(ln[1:])
  dict['data'].append(entry)
 return dict
dict=openFile(fn)
print(len(dict['data']))

def counter(num):
 if num == 5:
  num=1
 else:
  num+=1
 return num

def fiveFold(dict):
 a=[[],[],[],[],[]]
 n=[[],[],[],[],[]]
 d,r=1,1
 for e in dict['target']:
  a[d-1].append(e)
  d=counter(d)
 for w in dict['data']:
  n[r-1].append(w)
  r=counter(r)
 return (a,n)
splited=fiveFold(dict)
target,data=splited[0],splited[1]

def trainTest(target,data,num):
 x_test,y_test=np.array(data[num]),np.array(target[num])
 xtrain,ytrain=[],[]
 for a in range (5):
  if a+1 != num:
   for j in data[num]:
    xtrain.append(j)
   for i in target[num]:
    ytrain.append(i)
 x_train,y_train=np.array(xtrain),np.array(ytrain)
 return (x_test,y_test,x_train,y_train)
num=0
sp=trainTest(target,data,num)
x_test,y_test,x_train,y_train=sp[0],sp[1],sp[2],sp[3]

def MNB(x_test,y_test,x_train,y_train):
 from sklearn.naive_bayes import MultinomialNB
 mnb=MultinomialNB()
 y_pred=mnb.fit(x_train, y_train).predict(x_test)
 for i in range(len(y_test)):
  print(y_test[i],y_pred[i])
#MNB(x_test,y_test,x_train,y_train)

def LogisticRegression(x_test,y_test,x_train,y_train):
 from sklearn.linear_model import LogisticRegression
 #newton-cg does better than lbfgs on my input
 y_pred=LogisticRegression(random_state=0,solver='newton-cg',
         multi_class='multinomial').fit(x_train,y_train).predict(x_test)
 for i in range(len(y_test)):
  print(y_test[i],y_pred[i])
#LogisticRegression(x_test,y_test,x_train,y_train)

def KNN(x_test,y_test,x_train,y_train):
 from sklearn.neighbors import KNeighborsRegressor
 y_pred=KNeighborsRegressor(n_neighbors=10).fit(x_train,y_train).predict(x_test)
 for i in range(len(y_test)):
  print(y_test[i],y_pred[i])
#KNN(x_test,y_test,x_train,y_train)
 
def BayesRidge(x_test,y_test,x_train,y_train):
 from sklearn import linear_model
 BR=linear_model.BayesianRidge()
 y_pred=BR.fit(x_train,y_train).predict(x_test)
 for i in range (len(y_test)):
  print(y_test[i],y_pred[i])
#BayesRidge(x_test,y_test,x_train,y_train)

def SVM(x_test,y_test,x_train,y_train):
 from sklearn.svm import SVR
 y_pred=SVR(gamma=0.9,C=1.0,epsilon=0.2).fit(x_train,y_train).predict(x_test)
 for i in range (len(y_test)):
  print(y_test[i],y_pred[i])
#SVM(x_test,y_test,x_train,y_train)

def SGD(x_test,y_test,x_train,y_train):
 from sklearn import linear_model
 y_pred=linear_model.SGDRegressor().fit(x_train,y_train).predict(x_test)
 for i in range (len(y_test)):
  print(y_test[i],y_pred[i])
#SGD(x_test,y_test,x_train,y_train)
