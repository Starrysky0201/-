#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入自定义包load_image
from load_image import load_image_people
import numpy as np
import matplotlib.pylab as plt


# In[2]:


image_path = ".\\lfw_home\\lfw_funneled"
data, label, name_dict = load_image_people(image_path)


# In[3]:


n_classes = len(name_dict)
target_names = name_dict.values()


# In[4]:


# 特征选择
# F检验
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
data_p = data
data = SelectKBest(f_classif, k=7500).fit_transform(data_p, label)


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


# In[6]:


# 数据分离
X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=0.20, random_state=42)
# 降维维数3
n_components = 175


# In[7]:


# 定义SVM训练器
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)


# In[8]:


# KPCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
kpca = KernelPCA(n_components=n_components,kernel='cosine').fit(X_train)
X_train_kpca = kpca.transform(X_train)
X_test_kpca = kpca.transform(X_test)

clf.fit(X_train_kpca, y_train)
y_score_kpca = clf.decision_function(X_test_kpca)
y_pred = clf.predict(X_test_kpca)
# 预测
# print(classification_report(y_test, y_pred, target_names=target_names))
y_accuracy = accuracy_score(y_test, y_pred)
print(y_accuracy)


# In[ ]:




