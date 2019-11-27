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
  dict['target'].append(int(round(float(ln[0]),1)*10))  #classifier input
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
num=1
sp=trainTest(target,data,num)
x_test,y_test,x_train,y_train=sp[0],sp[1],sp[2],sp[3]

def MNB(x_test,y_test,x_train,y_train):
 from sklearn.naive_bayes import MultinomialNB
 mnb=MultinomialNB()
 y_pred=mnb.fit(x_train, y_train).predict(x_test)
 for i in range(len(y_test)):
  print(y_test[i],y_pred[i])
MNB(x_test,y_test,x_train,y_train)

def LogisticRegression(x_test,y_test,x_train,y_train):
 from sklearn.linear_model import LogisticRegression
 #newton-cg does better than lbfgs on my input
 y_pred=LogisticRegression(random_state=0,solver='newton-cg',
         multi_class='multinomial').fit(x_train,y_train).predict(x_test)
 for i in range(len(y_test)):
  print(y_test[i],y_pred[i])
LogisticRegression(x_test,y_test,x_train,y_train)
