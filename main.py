import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve,auc,roc_auc_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler,OrdinalEncoder
from pyod.models.auto_encoder import AutoEncoder
from tqdm import tqdm
from sklearn import metrics
import seaborn as sns
matplotlib.rcParams.update({'font.size': 30})
sns.set(style='whitegrid')
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)


trainData = pd.read_csv('./dayr4.2.csv')
# print(trainData.columns)
LISANCOLUMNS = [2,11]
DROPCOLUMNS = []
LABELCOLUMNS = [507]
FEATURECOLUMNS = [i for i in range(trainData.shape[1]) if i not in DROPCOLUMNS and i not in LABELCOLUMNS]
FENLI = False
MODELNAME = "AutoEncoder"
MINSECTIONNUM1 = 50
MINSECTIONNUM2 = 1000
HIDDEN_NEURONS = [256,128,64,32,16,8,4,4,8,16,32,64,128,256]
GLOBALEPOCH = 100
LOCALEPOCH = 100
classifiers = {
        "Isolation Forest":IsolationForest(n_estimators=100, max_samples=int(len(trainData)*0.01) ,bootstrap=False,warm_start=True,random_state=6666,
                                    max_features = 1.0, verbose=0,n_jobs=-1),
        "AutoEncoder": AutoEncoder(hidden_neurons=HIDDEN_NEURONS,epochs=GLOBALEPOCH,batch_size=1000,verbose=0)
        }
# print(trainData.head(1))

if FENLI:
    testData = pd.read_csv("kddTest+.csv",error_bad_lines=False,header=None)

columnName = [i for i in range(trainData.shape[1])]

trainData.columns = [i for i in range(trainData.shape[1])]
trainData = trainData.reset_index(drop=True)

clf=classifiers[MODELNAME]
train_X=trainData.iloc[:,:-1]
clf.fit(train_X)

def scoreShower(trueAll,predAll):
    contrast1 = pd.concat([predAll,trueAll],axis=1)
    contrast1.columns = ['pred','ret']
    contrast1.sort_values(by = 'ret',inplace=True)
    x = np.arange(predAll.shape[0])
    plt.style.use('bmh')
    fig1 = plt.figure(1,figsize=(20,6))
    plt.scatter(x,contrast1[['pred']],color = 'blue',marker = '8',s = 0.5,alpha=0.8)
    plt.scatter(x,contrast1[['ret']],color = 'red',marker = 'o',s = 0.7)
    plt.show()

if FENLI:
    test_x = testData.iloc[:,:-1]
else:
    test_x = trainData.iloc[:,:-1]
if MODELNAME=='Isolation Forest':
    scores_prediction = -1*pd.DataFrame(clf.score_samples(test_x))
else:
    scores_prediction = pd.DataFrame(clf.decision_function(test_x))

scoreDfScaler = MinMaxScaler()
scores_prediction_trans = pd.DataFrame(scoreDfScaler.fit_transform(scores_prediction))
print(scores_prediction_trans.describe())


score_insider = pd.concat([scores_prediction_trans,train_X[2]],axis=1)
score_insider.columns = ['score_pre','userID']
user_insider = pd.DataFrame(score_insider.groupby('userID')['score_pre'].mean())
user_insider.sort_values(by='score_pre',inplace=True,ascending=False)
insiderRes = user_insider.index


score_insider2 = pd.concat([scores_prediction_trans,train_X[11]],axis=1)
score_insider2.columns = ['score_pre','teamID']
user_insider2 = pd.DataFrame(score_insider2.groupby('teamID')['score_pre'].mean())
user_insider2.sort_values(by='score_pre',inplace=True,ascending=False)
insiderRes2 = user_insider2.index

if FENLI:
    scores_true = testData.iloc[:,-1:].reset_index(drop=True)
else:
    scores_true = trainData.iloc[:,-1:].reset_index(drop=True)

scores_true[507] = scores_true[507].apply(lambda x:0 if x==0 else 1)
print(scores_true)
print(scores_true.describe())

def local(train_X,test_X_0,columnName,nowName):
    if nowName=='2':
        MINSECTIONNUM = MINSECTIONNUM1
    else:
        MINSECTIONNUM = MINSECTIONNUM2
    test_X = test_X_0.copy()
    test_X['score'] = None
    featureName = [i for i in columnName if i !=nowName]
    smallSectionIDs = train_X[nowName].unique()
    if len(smallSectionIDs)==1:
        return None
    for sectionID in tqdm(smallSectionIDs):
        
        section = train_X.loc[train_X[nowName]==sectionID].reset_index(drop=True)
        section = section[featureName]
        if section.shape[0]<MINSECTIONNUM:
            continue
        section.columns = [str(i) for i in range(section.shape[1])]
        classifiers2 = {
            "Isolation Forest":IsolationForest(n_estimators=100, max_samples=int(len(section)*0.5) ,bootstrap=False,warm_start=True,random_state=6666,
                                    max_features = 1.0, verbose=0,n_jobs=-1),
            "AutoEncoder": AutoEncoder(hidden_neurons=HIDDEN_NEURONS,epochs=LOCALEPOCH,batch_size=1000,verbose=0)}

        
        max_samples = max(1,int(len(section)*0.01))
        clf2=classifiers2[MODELNAME]
        clf2.fit(section)
        #display(section.head())
        testSection = test_X  .loc[test_X[nowName]==sectionID].reset_index(drop=True)

        testSection = testSection[featureName]
        testSection.columns = [str(i) for i in range(testSection.shape[1])]
        #print('='*80,section.shape[1])
        #print('='*80,testSection.shape[1])
        #display(testSection.head())
        if MODELNAME=='Isolation Forest':
            scores_prediction = -1*pd.DataFrame(clf2.score_samples(testSection))
        else:
            scores_prediction = pd.DataFrame(clf2.decision_function(testSection))

        scoreDfScaler = MinMaxScaler()
        scores_prediction_trans = pd.DataFrame(scoreDfScaler.fit_transform(scores_prediction))
        test_X.loc[test_X[nowName]==sectionID,'score']=scores_prediction_trans.values
        #.values很关键
        #print(test_X.loc[test_X[nowName]==sectionID]['score'])
    return test_X['score']

del trainData

sevenLocalScore = pd.DataFrame([])
train_X.columns = [str(i) for i in range(train_X.shape[1])]
test_x.columns = [str(i) for i in range(test_x.shape[1])]
LISANCOLUMNS = [str(i) for i in LISANCOLUMNS]
for i in LISANCOLUMNS:
    print('='*80)
    print('当前处理变量：'+i)
    nowLocalScore = local(train_X,test_x,train_X.columns,i)
    sevenLocalScore = pd.concat([sevenLocalScore,nowLocalScore],axis=1)
    del nowLocalScore
print(sevenLocalScore) 

sevenLocalScore.columns = [str(i)+'_local' for i in LISANCOLUMNS]
sevenLocalScore.head(10)
print(sevenLocalScore)

conbinedScore = pd.concat([train_X['2'],train_X['11'],sevenLocalScore,scores_prediction_trans],axis=1)
print(conbinedScore)

def tellMean(x,K,L):
    res = [x[0]]
    len1 = len(insiderRes)
    len2 = len(insiderRes2)
    if x['2'] in insiderRes[:int(K*len1)] and x['2_local']!=None:
        res.append(x['2_local'])
    if x['11'] in insiderRes2[:int(L*len2)] and x['11_local']!=None:
        res.append(x['11_local'])
    return np.mean(res)

def tellMax(x,K,L):
    res = [x[0]]
    len1 = len(insiderRes)
    len2 = len(insiderRes2)
    if x['2'] in insiderRes[:int(K*len1)] and x['2_local']!=None:
        res.append(x['2_local'])
    if x['11'] in insiderRes2[:int(L*len2)] and x['11_local']!=None:
        res.append(x['11_local'])
    return np.max(res)

conbinedScore['conditionCombinedMean'] = conbinedScore.apply(tellMean,args=(0.3,0.2),axis=1)
conbinedScore['conditionCombinedMax'] = conbinedScore.apply(tellMax,args=(0.3,0.2),axis=1)

# import pickle
# # pickle a variable to a file
# file = open('ISconbinedScore.pickle', 'wb')
# pickle.dump(conbinedScore, file)
# file.close()

plt.figure(figsize=(8,6),dpi=400) # 设置分辨率
fpr,tpr,thresholds=metrics.roc_curve(scores_true,scores_prediction_trans)
# print(scores_prediction_trans)
scores_prediction_class = [1 if i>0.5 else 0 for i in scores_prediction_trans[0]]
# print(scores_true)
# print(scores_prediction_class)
accuracy = metrics.accuracy_score(scores_true,scores_prediction_class)
tp,fp,fn,tn = metrics.confusion_matrix(scores_true,scores_prediction_class).ravel()
print(tp,fp,fn,tn)
precision = metrics.precision_score(scores_true,scores_prediction_class, average='binary')
recall = metrics.recall_score(scores_true,scores_prediction_class, average='binary')
f1 = metrics.f1_score(scores_true,scores_prediction_class, average='binary')
sns.lineplot(x=fpr*100,y=tpr*100, markers="o")
auc_score = roc_auc_score(scores_true,scores_prediction_trans)

fpr2,tpr2,thresholds2=metrics.roc_curve(scores_true,conbinedScore['conditionCombinedMean'])
conbinedScore_class = [1 if i>0.5 else 0 for i in conbinedScore['conditionCombinedMean']]
accuracy2 = metrics.accuracy_score(scores_true,conbinedScore_class)
precision2 = metrics.precision_score(scores_true,conbinedScore_class, average='binary')
recall2 = metrics.recall_score(scores_true,conbinedScore_class, average='binary')
f1_2 = metrics.f1_score(scores_true,conbinedScore_class, average='binary')
auc_score2 = roc_auc_score(scores_true,conbinedScore['conditionCombinedMean'])
auc_score2
sns.lineplot(x=fpr2*100,y=tpr2*100)

fpr3,tpr3,thresholds3=metrics.roc_curve(scores_true,conbinedScore['conditionCombinedMax'])
conbinedScore_class3 = [1 if i>0.5 else 0 for i in conbinedScore['conditionCombinedMax']]
accuracy3 = metrics.accuracy_score(scores_true,conbinedScore_class3)
precision3 = metrics.precision_score(scores_true,conbinedScore_class3, average='binary')
recall3 = metrics.recall_score(scores_true,conbinedScore_class3, average='binary')
f1_3 = metrics.f1_score(scores_true,conbinedScore_class3, average='binary')
auc_score3 = roc_auc_score(scores_true,conbinedScore['conditionCombinedMax'])
auc_score3
sns.lineplot(x=fpr3*100,y=tpr3*100)

print('IS, Accuracy = {:.3f}, Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}, AUC = {:.3f}'.format(accuracy,precision,recall,f1,auc_score))
print('Fusion-IS(MEAN), Accuracy = {:.3f}, Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}, AUC = {:.3f}'.format(accuracy2,precision2,recall2,f1_2,auc_score2))
print('Fusion-IS(MAX), Accuracy = {:.3f}, Precision = {:.3f}, Recall = {:.3f}, F1 = {:.3f}, AUC = {:.3f}'.format(accuracy3,precision3,recall3,f1_3,auc_score3))

x = np.linspace(0, 100, 1000)
y = x
plt.plot(x,y,ls='--',lw=0.5,c='k')

plt.xlabel('False positive rate %',fontsize=15)
plt.ylabel('True positive rate %',fontsize=15)

plt.legend(['IS, AUC = {:.3f}'.format(auc_score),
    'Fusion-IS(MEAN), AUC = {:.3f}'.format(auc_score2),
    'Fusion-IS(MAX), AUC = {:.3f}'.format(auc_score3),

],fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('IS.pdf',dpi=2000)
plt.show()
#ae,max,0.77