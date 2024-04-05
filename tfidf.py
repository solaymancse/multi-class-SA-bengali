
#As we dont have any GPU we will be using google collab for GPU usage
#Mout at drive 
import sys
from google.colab import drive
from pathlib import Path
drive.mount("/content/drive", force_remount=True)

#useful downloads
!pip install bnlp_toolkit
!pip install chart-studio

# Commented out IPython magic to ensure Python compatibility.
#import packages

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sn
# %matplotlib inline
import re
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn import preprocessing
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn.model_selection import train_test_split
from chart_studio import plotly as py
#import plotly.plotly as py
import plotly.offline as pyo
pyo.init_notebook_mode()
#from plotly.offline import init_notebook_mode
#init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from bnlp.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

#read the data
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/dataset/bangla_comments_tokenized.csv')
df.tail()

#rename  'not bully' to 'acceptable'
df['label'] = df['label'].replace({'not bully':'acceptable'})
df.head()

#fill na values with zero
df = df.fillna(0)
df.head()

# label encoding for output

sample_data = [2000,5000,10000,20000,30000,40000]

def label_encoding(category,bool):
  le = preprocessing.LabelEncoder()
  le.fit(category)
  encoded_labels = le.transform(category)
  labels = np.array(encoded_labels) # Converting into numpy array
  class_names =le.classes_ ## Define the class names again
  if bool == True:
    print("\n\t\t\t Label Encoding ","\nClass Names:",le.classes_)
    for i in sample_data:
      print(category[i],' ', encoded_labels[i],'\n')
    return labels

df.labels = label_encoding(df.label,True)

#confusion matrix

def conf_matrix(pred,classfier,directory,filename):
  predictions = pred
  y_pred = np.array(predictions)
  cm = confusion_matrix(y_test, y_pred) 
# Transform to df for easier plotting
  
  cm_df = pd.DataFrame(cm,
                       index = ['Political', 'acceptable', 'religious', 'sexual'], 
                       columns = ['Political', 'acceptable', 'religious', 'sexual'])
  plt.figure(figsize=(8,6))
  sn.heatmap(cm_df, annot=True,cmap="YlGnBu", fmt='g')
  plt.title('\n'+classfier+'Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_pred)*100))
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.xticks(rotation = 45)
  plt.yticks(rotation = 45)
  plt.savefig("/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/result_analysis/"+directory+"/"+filename+".png")
  plt.show()
  plt.close()

#report generation

def report_generate(pred,classfier,directory,filename):
  report = pd.DataFrame(classification_report(y_true = y_test, y_pred = pred, output_dict=True)).transpose()
  report = report.rename(index={'0': 'political','1':'acceptable','2':'religious','3':'sexual'})
  report[['precision','recall','f1-score']]=report[['precision','recall','f1-score']].apply(lambda x: round(x*100,2))
  report=report.drop(["support"],axis=1)
  columns = ['precision','recall','f1-score']
  report.columns = columns
  plt = report.plot(kind='bar',figsize=(12,6))
  
  plot=plt.tick_params(rotation=40)
  plt.figure.savefig("/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/result_analysis/"+directory+"/"+filename+".png")
  
  return plot,report

#plot comaparison 
def compare_plots(y_value,directory,filename):
  plt.subplots(figsize=(11,8))
  sn.barplot(x="Name", y=y_value ,data=compare,palette='hot',edgecolor=sn.color_palette('dark',7))
  plt.xticks(rotation=45)
  plt.title('Comparing techniques with '+y_value+'.')
  plt.savefig("/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/result_analysis/"+directory+"/"+filename+".png")
  plt.show()
  return plt

#divide the model for trainning and testing
df.text=df.text.apply(str)
X = df.text.values
y = df.labels
#categories = ['label_sexual','label_religious','label_troll','label_threat','label_acceptable']  #targeted labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

#checking...
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""**Decision Tree model with Tfidf using pipeline**"""

TFIDF_DCT_pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=15000, min_df=5,ngram_range=(1,2))),
                
                ('clf', OneVsRestClassifier(DecisionTreeClassifier(criterion='entropy', min_samples_split=40, min_samples_leaf=5, random_state=42))),
            ])

TFIDF_DCT_pipeline.fit(X_train, y_train)
TFIDF_DCT_pipeline_prediction = TFIDF_DCT_pipeline.predict(X_test)

# save the model to disk
path = '/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/model/'
filename = path + 'decision_tree_model.sav'
joblib.dump(TFIDF_DCT_pipeline_prediction, open(filename, 'wb'))

# load the model from disk
loaded_model = joblib.load(open(filename, 'rb'))
DCT = accuracy_score(loaded_model, y_test)
print(DCT)
print('accuracy %s' % accuracy_score(TFIDF_DCT_pipeline_prediction, y_test))

"""**MultinomialNB model with Tfidf using pipeline**"""

# Define a pipeline combining a text feature extractor with multi lable classifier
TFIDF_NB_pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=9000, min_df=5,ngram_range=(1,2))),
                
                ('clf', OneVsRestClassifier(MultinomialNB(
                    alpha=0.15,fit_prior=True, class_prior=None))),
            ])

TFIDF_NB_pipeline.fit(X_train, y_train)
TFIDF_NB_pipeline_prediction = TFIDF_NB_pipeline.predict(X_test)

filename = path + 'MNB.sav'
joblib.dump(TFIDF_NB_pipeline_prediction, open(filename, 'wb'))

# load the model from disk
loaded_model = joblib.load(open(filename, 'rb'))
NB = accuracy_score(loaded_model, y_test)
print(NB)

print('accuracy %s' % accuracy_score(TFIDF_NB_pipeline_prediction, y_test))

NB_report=report_generate(TFIDF_NB_pipeline_prediction,"NaiveBayes","MultiNB","multiNB_report")
NB_report[1]

conf_matrix(TFIDF_NB_pipeline_prediction,"NaiveBayes","MultiNB","multinb_confusion")

"""**SGDclassifier model with TFIDF using pipeline**"""

TFIDF_SGD_pipeline = Pipeline([
                               
                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=13000,min_df=1,ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, random_state=42, max_iter=200, tol=None)))
            ])


TFIDF_SGD_pipeline.fit(X_train, y_train)
TFIDF_SGD_pipeline_prediction = TFIDF_SGD_pipeline.predict(X_test)

filename = path + 'SGD.sav'
joblib.dump(TFIDF_SGD_pipeline_prediction, open(filename, 'wb'))

# SGD = accuracy_score(TFIDF_SGD_pipeline_prediction, y_test)
#SGD_f1 = f1_score(TFIDF_SGD_pipeline_prediction, y_test)

print('accuracy %s' % accuracy_score(TFIDF_SGD_pipeline_prediction, y_test))

SGD_report=report_generate(TFIDF_SGD_pipeline_prediction,"SGD classifier","SGD","SGD_report")
SGD_report[1]

conf_matrix(TFIDF_SGD_pipeline_prediction,"SGD classifier","SGD","SGD_confusion")

"""**Logistic Regression model with TFIDF using pipeline**"""

TFIDF_LR_pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=15000,min_df=5,ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(LogisticRegression(multi_class='ovr',solver='liblinear',C=1,random_state=42,tol=0.0001,max_iter=200)))
            ])

TFIDF_LR_pipeline.fit(X_train, y_train)
TFIDF_LR_pipeline_prediction = TFIDF_LR_pipeline.predict(X_test)

filename = path + 'LR.sav'
joblib.dump(TFIDF_LR_pipeline_prediction, open(filename, 'wb'))

# LR = accuracy_score(TFIDF_LR_pipeline_prediction, y_test)
#LR_f1 = f1_score(TFIDF_LR_pipeline_prediction, y_test)

print('accuracy %s' % accuracy_score(TFIDF_LR_pipeline_prediction, y_test))

conf_matrix(TFIDF_LR_pipeline_prediction,"Logistic regressor classifier","Logistic Regression","LR_confusion")

LR_report=report_generate(TFIDF_LR_pipeline_prediction,"Logistic regressor classifier","Logistic Regression","LR_report")
LR_report[1]

"""**RandomforrestClassifier model with TFIDF using pipeline**"""

from sklearn.ensemble import RandomForestClassifier
TFIDF_DT_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=15000,min_df=5,ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=200,criterion ='entropy')))
            ])


TFIDF_DT_pipeline.fit(X_train, y_train)
TFIDF_DT_pipeline_prediction = TFIDF_DT_pipeline.predict(X_test)
filename = path + 'RF.sav'
joblib.dump(TFIDF_DT_pipeline_prediction, open(filename, 'wb'))

# DT = accuracy_score(TFIDF_DT_pipeline_prediction, y_test)
#DT_f1 = f1_score(TFIDF_DT_pipeline_prediction, y_test)

print('accuracy %s' % accuracy_score(TFIDF_DT_pipeline_prediction, y_test))

conf_matrix(TFIDF_DT_pipeline_prediction,"RandForrest classifier","Random forrest","RF_confusion")

DT_report=report_generate(TFIDF_DT_pipeline_prediction,"RandForrest classifier","Random forrest","RF_report")
DT_report[1]

"""**SVC**"""

#, gamma=0.001, C=1000
TFIDF_SVC_pipeline = Pipeline([
                               
                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=8000,min_df=5,ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(SVC(random_state=42)))
            ])


TFIDF_SVC_pipeline.fit(X_train, y_train)
TFIDF_SVC_pipeline_prediction = TFIDF_SVC_pipeline.predict(X_test)
filename = path + 'SVC.sav'
joblib.dump(TFIDF_SVC_pipeline_prediction, open(filename, 'wb'))

# SVC = accuracy_score(TFIDF_SVC_pipeline_prediction, y_test)
 #SVC_f1 = f1_score(TFIDF_SVC_pipeline_prediction, y_test)
print('accuracy %s' % accuracy_score(TFIDF_SVC_pipeline_prediction, y_test))

conf_matrix(TFIDF_SVC_pipeline_prediction,"SVC classifier","SVC","SVC_confusion")

SVC_report=report_generate(TFIDF_SVC_pipeline_prediction,"SVC classifier","SVC","SVC_report")
SVC_report[1]

"""**Result Comparison**"""

# Functions to compute True Positives, True Negatives, False Positives and False Negatives

def true_positive(y_true, y_pred):   
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp

def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn

def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp

def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1

#Computation of macro-averaged precision

def macro_precision(y_true, y_pred):
    # find the number of classes
    num_classes = len(np.unique(y_true))
    # initialize precision to 0
    precision = 0
    categories = 4
    # loop over all classes
    for class_ in range (categories):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # compute true positive for current class
        tp = true_positive(temp_true, temp_pred)
        # compute false positive for current class
        fp = false_positive(temp_true, temp_pred)
        # compute precision for current class
        temp_precision = tp / (tp + fp + 1e-6)
        # keep adding precision for all classes
        precision += temp_precision
        
    # calculate and return average precision over all classes
    precision /= num_classes
    return precision

MLA = {
     'Naive Bayes' : TFIDF_NB_pipeline,
     'Decision Tree' : TFIDF_DCT_pipeline,
     'SGD Classifier' : TFIDF_SGD_pipeline,
     'Logistic Regression' : TFIDF_LR_pipeline,
     'Random Forrest' : TFIDF_DT_pipeline,
     'SVC' : TFIDF_SVC_pipeline  
}

columns = []
compare = pd.DataFrame(columns = columns)
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes=4
row_index = 0
for name,alg in MLA.items():
    #fp, tp, th = roc_curve(y_test, predicted ,pos_label=['Political', 'acceptable', 'religious', 'sexual'])
    MLA_name = name
    #alg.fit(X_train, y_train)
    predicted = alg.predict(X_test)
    micro_averaged_recall = recall_score(y_test, predicted, average = 'micro')
    macro_averaged_f1 = f1_score(y_test, predicted, average = 'macro')
    
    compare.loc[row_index,'Name'] = MLA_name
    #compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 4)
    compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    compare.loc[row_index, 'Precision'] = macro_precision(y_test, predicted)
    compare.loc[row_index, 'Recall'] = micro_averaged_recall
    compare.loc[row_index, 'F1 Score'] = macro_averaged_f1
    row_index+=1
    
compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
compare

# train_comparison=compare_plots("Train Accuracy","ML comparison","train comparison")
compare.to_csv('/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/result_analysis/compareResult.csv')

"""**K-Fold**"""

from sklearn.ensemble import RandomForestClassifier
TFIDF_DCT_pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=15000, min_df=5,ngram_range=(1,2))),
                
                ('clf', OneVsRestClassifier(DecisionTreeClassifier(criterion='entropy', min_samples_split=40, min_samples_leaf=5, random_state=42))),
            ])
TFIDF_NB_pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=9000, min_df=5,ngram_range=(1,2))),
                
                ('clf', OneVsRestClassifier(MultinomialNB(
                    alpha=0.15,fit_prior=True, class_prior=None))),
            ])
TFIDF_SGD_pipeline = Pipeline([
                               
                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=13000,min_df=1,ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, random_state=42, max_iter=200, tol=None)))
            ])
TFIDF_LR_pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=15000,min_df=5,ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(LogisticRegression(multi_class='ovr',solver='liblinear',C=1,random_state=42,tol=0.0001,max_iter=200)))
            ])
TFIDF_DT_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=15000,min_df=5,ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=200,criterion ='entropy')))
            ])
TFIDF_SVC_pipeline = Pipeline([
                               
                ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(),lowercase=False,max_features=8000,min_df=5,ngram_range=(1,2))),
                ('clf', OneVsRestClassifier(SVC(random_state=42)))
            ])

from sklearn.model_selection import cross_val_score
n_folds = 10
cv_score_NB = cross_val_score(estimator=TFIDF_NB_pipeline, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)
cv_score_DT = cross_val_score(estimator=TFIDF_DCT_pipeline, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)
cv_score_SGD = cross_val_score(estimator=TFIDF_SGD_pipeline, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)
cv_score_LR = cross_val_score(estimator=TFIDF_LR_pipeline, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)
cv_score_RF = cross_val_score(estimator=TFIDF_DT_pipeline, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)
cv_score_SVC = cross_val_score(estimator=TFIDF_SVC_pipeline, X=X_train, y=y_train, cv=n_folds, n_jobs=-1)

cv_result = {'MNB': cv_score_NB, 'DT': cv_score_DT, 'SGD': cv_score_SGD, 'LR': cv_score_LR, 'RF': cv_score_RF, 'SVC': cv_score_SVC}
cv_data = {model: [score.mean(), score.std()] for model, score in cv_result.items()}
cv_df = pd.DataFrame(cv_data, index=['Mean_accuracy', 'Variance'])
cv_df

cv_df.to_csv('/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/result_analysis/vardata.csv')

plt.figure(figsize=(12,8))
n_folds=10
plt.plot(cv_result['SVC'],marker='o')
plt.plot(cv_result['SGD'],marker='o')
plt.plot(cv_result['RF'],marker='o')
plt.plot(cv_result['LR'],marker='o')
plt.plot(cv_result['MNB'],marker='o')
plt.plot(cv_result['DT'],marker='o')
plt.title('CV score for each fold',fontsize=22)
plt.ylabel('Accuracy',fontsize=18)
plt.xlabel('Trained fold',fontsize=18)
plt.xticks([k for k in range(n_folds)])
plt.tick_params(axis='x',rotation=0,labelsize=25)
plt.tick_params(axis='y',rotation=0,labelsize=25)
plt.legend(['SVC', 'SGD', 'RF', 'LR', 'MNB', 'DT'], loc=2,bbox_to_anchor = (1,1), prop={'size': 20})
plt.savefig("/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/result_analysis/ML comparison/varience.png")
plt.show()

plt.figure(figsize=(20,8))
n_folds=10
plt.plot(cv_result['SVC'],marker='o')
plt.plot(cv_result['SGD'],marker='o')
plt.plot(cv_result['RF'],marker='o')
plt.plot(cv_result['LR'],marker='o')
plt.plot(cv_result['MNB'],marker='o')
plt.plot(cv_result['DT'],marker='o')
plt.title('CV score for each fold',fontsize=20)
plt.ylabel('Accuracy',fontsize=18)
plt.xlabel('Trained fold',fontsize=18)
plt.xticks([k for k in range(n_folds)])
plt.tick_params(axis='x',rotation=0,labelsize=25)
plt.tick_params(axis='y',rotation=0,labelsize=25)
plt.legend(['SVC', 'SGD', 'RF', 'LR', 'MNB', 'DT'], loc=2,bbox_to_anchor = (1,1), prop={'size': 20})
# plt.savefig("/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/result_analysis/ML comparison/varience.png")
plt.show()

#read the data
compare=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/result_analysis/compareResult.csv')
compare

compare = compare.rename(columns={'Name': 'Machine Learning Algorithms'})

compare

#plot comaparison 
def compare_plots(y_value,directory,filename, score):
  plt.subplots(figsize=(11,8))
  # color_palette("vlag", as_cmap=True)
  sn.barplot(x="Machine Learning Algorithms", y=y_value,palette='icefire',data=compare,edgecolor=sn.color_palette('dark',7))
  
  plt.tick_params(axis='x',rotation=0,labelsize=22)
  plt.tick_params(axis='y',rotation=0,labelsize=18)
  plt.title('Comparing performance with '+y_value+'.', fontsize=20)
  plt.ylabel(score +' Score', fontsize=22)
  plt.xlabel('Machine Learning Algorithms', fontsize=16)
  plt.savefig("/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/result_analysis/"+directory+"/"+filename+".png")
  plt.show()
  return plt

test_comparison=compare_plots("Test Accuracy","ML comparison","test comparison","Accuracy")

Precision_comparison=compare_plots("Precision","ML comparison","Precision comparison", 'Precision')

F1_comparison=compare_plots("F1 Score","ML comparison","F1 Score comparison","F1")

Recall_comparison=compare_plots("Recall","ML comparison","Recall comparison","Recall")