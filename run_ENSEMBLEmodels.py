import sys
import numpy as np
fn=sys.argv[1]
models=sys.argv[2]

def genEntry(L):
 entry=[]
 for i in L:
  entry.append(float(i))
 return entry

def openFile(fn):
 file=open(fn,"r")
 file.readline()
 dict={'data':[],'target':[]}
 for ln in file:
  ln=ln.strip("\r\n").split(",")
  dict['target'].append(float(ln[0]))  #regressor input
  entry=genEntry(ln[1:])
  dict['data'].append(entry)
 return dict
dict=openFile(fn)

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

def outFile(y_test,y_pred,model,outfile):
 s=0
 for i in range(len(y_test)):
  outfile.write(str(y_test[i])+","+str(y_pred[i])+"\r\n")
  s+=(y_test[i]-y_pred[i])**2
 MSE=s/len(y_test)
 print(MSE)
 
def MNB(x_test,y_test,x_train,y_train):
 from sklearn.naive_bayes import MultinomialNB
 mnb=MultinomialNB()
 y_pred=mnb.fit(x_train, y_train).predict(x_test)
 return y_pred

def LogisticR(x_test,y_test,x_train,y_train):
 from sklearn.linear_model import LogisticRegression
 #newton-cg does better than lbfgs on my input
 y_pred=LogisticRegression(random_state=0,solver='newton-cg',
         multi_class='multinomial').fit(x_train,y_train).predict(x_test)
 return y_pred

def KNN(x_test,y_test,x_train,y_train):
 from sklearn.neighbors import KNeighborsRegressor
 y_pred=KNeighborsRegressor(n_neighbors=10).fit(x_train,y_train).predict(x_test)
 return y_pred
 
def BayesRidge(x_test,y_test,x_train,y_train):
 from sklearn import linear_model
 BR=linear_model.BayesianRidge()
 y_pred=BR.fit(x_train,y_train).predict(x_test)
 return y_pred

def SVM(x_test,y_test,x_train,y_train):
 from sklearn.svm import SVR
 y_pred=SVR(gamma=0.9,C=1.0,epsilon=0.2).fit(x_train,y_train).predict(x_test)
 return y_pred

def SGD(x_test,y_test,x_train,y_train):
 from sklearn import linear_model
 y_pred=linear_model.SGDRegressor().fit(x_train,y_train).predict(x_test)
 return y_pred

def ExtremeRanTree(x_test,y_test,x_train,y_train):
 from sklearn.ensemble import ExtraTreesRegressor
 y_pred=ExtraTreesRegressor(n_estimators=100).fit(x_train,y_train).predict(x_test)
 return y_pred

def AdaBoostR(x_test,y_test,x_train,y_train):
 from sklearn.ensemble import AdaBoostRegressor
 y_pred=AdaBoostRegressor().fit(x_train,y_train).predict(x_test)
 return y_pred

def GradientBoostR(x_test,y_test,x_train,y_train):
 from sklearn.ensemble import GradientBoostingRegressor
 y_pred=GradientBoostingRegressor().fit(x_train,y_train).predict(x_test)
 return y_pred

###########################use python3 to run these, where sklearn version == 0.21.3###########################
def GaussianProcessR(x_test,y_test,x_train,y_train):
 from sklearn.gaussian_process import GaussianProcessRegressor
 from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
 kernel=DotProduct()+WhiteKernel()
 y_pred=GaussianProcessRegressor(kernel=kernel,random_state=0).fit(x_train,y_train).predict(x_test)
 return y_pred

def DecisionTreeR(x_test,y_test,x_train,y_train):
 from sklearn.model_selection import cross_val_score
 from sklearn.tree import DecisionTreeRegressor
 y_pred=DecisionTreeRegressor(random_state=0).fit(x_train,y_train).predict(x_test)
 return y_pred

def RandomForest(x_test,y_test,x_train,y_train):
 from sklearn.ensemble import RandomForestRegressor
 y_pred=RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100).fit(x_train,y_train).predict(x_test)
 return y_pred

def MLPerceptronR(x_test,y_test,x_train,y_train):
 from sklearn.neural_network import MLPRegressor
 y_pred=MLPRegressor().fit(x_train,y_train).predict(x_test)
 return y_pred

def ensembleModels(models,target,data):
 modelL={'MNB':MNB,'LogisticR':LogisticR,'KNN':KNN,'BayesRidge':BayesRidge,'SVM':SVM,'SGD':SGD,
         'ExtremeRanTree':ExtremeRanTree,'AdaBoostR':AdaBoostR,'GradientBoostR':GradientBoostR,
         'GaussianProcessR':GaussianProcessR,'DecisionTreeR':DecisionTreeR,'RandomForest':RandomForest,
         'MLPerceptronR':MLPerceptronR}
 models=models.split(",")
 ofn="ENSB"
 for model in models:
  ofn+="_"+model
 ofn+=".csv"
 outfile=open(ofn,"w")
 for a in range (5): 
  ypL, s=[],0
  i=trainTest(target,data,a)
  x_test,y_test,x_train,y_train=i[0],i[1],i[2],i[3]
  for model in models:
   ypL.append(modelL[model](x_test,y_test,x_train,y_train))
  for a in range(len(ypL[0])):
   com_pred, row=0, str(y_test[a])+","
   for i in range (len(models)):
    com_pred+=ypL[i][a]
    row+=str(ypL[i][a])+","
   com_pred=com_pred/len(models)
   row+=str(com_pred)+"\r\n"
   s+=(y_test[a]-com_pred)**2
   outfile.write(row)
  MSE=s/2916.0
  print MSE 
 outfile.close()  
ensembleModels(models,target,data)
