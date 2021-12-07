### with all paras

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("datanew.csv", header=0)
# print(df)
X1 = df.iloc[:, 1]
X2 = df.iloc[:, 2]
X3 = df.iloc[:, 3]
X4 = df.iloc[:, 4]
X5 = df.iloc[:, 5]
X6 = df.iloc[:, 6]
X7 = df.iloc[:, 7]
X8 = df.iloc[:, 8]
X9 = df.iloc[:, 9]
X10 = df.iloc[:, 10]
X11 = df.iloc[:, 11]
X12 = df.iloc[:, 12]
X13 = df.iloc[:, 13]
X14 = df.iloc[:, 14]
X15 = df.iloc[:, 15]
X16 = df.iloc[:, 16]
X17 = df.iloc[:, 17]
X18 = df.iloc[:, 18]
X = np.column_stack((X1, X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18))
# print(X)
# bad try:
# r = range(1,19)
# X = df.iloc[:,r]
y = df.iloc[:, 19]
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# ################### Ridge

# ridge = Ridge()
# alpha_can = np.logspace(-3, 2, 10)  #这是验证的参数列表
# rig = GridSearchCV(ridge, param_grid={'alpha': alpha_can}, cv=5)
# rig.fit(X_train, y_train)
# print ('best_alpha：', rig.best_params_)


#################### KNN

# parameters = {'n_neighbors': range(1,100)}
# knn = KNeighborsClassifier()
# clf = GridSearchCV(knn, parameters, cv=5)
# clf.fit(X, y)
# print("best_accuracy：%f" % clf.best_score_, "n_neighbors：", clf.best_params_)



##################### Random forest

# 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
# score_lt = []
#
# 每隔10步建立一个随机森林，获得不同n_estimators的得分
# for i in range(0,200,10):
# for i in range(50,70):
#     rfc = RandomForestClassifier(n_estimators=i+1
#                                 ,random_state=90)
#     score = cross_val_score(rfc, X, y, cv=10).mean()
#     score_lt.append(score)
# score_max = max(score_lt)
# # print('Max_score：{}'.format(score_max),
# #       'n_estimators：{}'.format(score_lt.index(score_max)*10+1))
# print('MaxScore：{}'.format(score_max),
#       'N_estimators：{}'.format(score_lt.index(score_max)+50))
#
# # x = np.arange(1,201,10)
# x = np.arange(50,70)
# plt.subplot(111)
# plt.plot(x, score_lt, 'r-')
# plt.show()

# 打印1-200，最大得分：0.845839788367107 子树数量为：61
# 打印50-69，最大得分：0.846234871014593 子树数量为：64 提高0.001
# n_estimators: 64



# 用网格搜索调整max_depth
# param_grid = {'max_depth':np.arange(1,20)}
# GS = GridSearchCV(rfc, param_grid, cv=10)
# GS.fit(X, y)
#
# best_param = GS.best_params_
# best_score = GS.best_score_
# print(best_param, best_score)

# {'max_depth': 14} 0.8450520437098001 还低了0.001 ？？ 离谱啊




# ################### baseline

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
ydummy = dummy.predict(X_test)
# print("Accuracy(Baseline):", accuracy_score(y_test, ydummy))

################### Models


# model_R = Ridge(alpha=0.1668100537200059).fit(X_train, y_train)
# model_K = KNeighborsClassifier(n_neighbors=21, weights='uniform').fit(X_train,y_train)
# model_F = RandomForestClassifier(n_estimators=64, random_state=90).fit(X_train, y_train)
model_R = Ridge(alpha=27.825594022071257).fit(X_train, y_train)
model_K = KNeighborsClassifier(n_neighbors=69, weights='uniform').fit(X_train,y_train)
model_F = RandomForestClassifier(n_estimators=92, random_state=90).fit(X_train, y_train)
model_S = SVC(kernel='linear', C=0.01, probability=True).fit(X_train, y_train)

# ################## ROC
fpr_F, tpr_F, _ = roc_curve(y_test, model_F.predict_proba(X_test)[:, 1])
# fpr_R, tpr_R, _ = roc_curve(y_test, model_R.predict_proba(X_test)[:, 1])
fpr_K, tpr_K, _ = roc_curve(y_test, model_K.predict_proba(X_test)[:, 1])
fpr_B, tpr_B, _ = roc_curve(y_test, dummy.predict_proba(X_test)[:, 1])


# ################### AUC
roc_auc_K = auc(fpr_K, tpr_K)
print("ROC_KNN:",roc_auc_K)
roc_auc_B = auc(fpr_B,tpr_B)
print("ROC_baseline:",roc_auc_B)
roc_auc_F = auc(fpr_F,tpr_F)
print("ROC_Randomforest:",roc_auc_F)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr_K, tpr_K, 'b', label ='KNN AUC = %0.2f' % roc_auc_K)
plt.plot(fpr_B, tpr_B, 'r', label ='Baseline AUC = %0.2f' % roc_auc_B)
plt.plot(fpr_F, tpr_F, 'y', label ='Random Forest AUC = %0.2f' % roc_auc_F)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve & AUC')
plt.show()

##################### Score

print("Score_Ridge:", model_R.score(X_test, y_test))
print("Score_KNN:",model_K.score(X_test,y_test))
print("Score_RandomForest:",model_F.score(X_test,y_test))
print("Score_baseline:",dummy.score(X_test,y_test))