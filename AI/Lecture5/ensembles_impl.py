

import warnings
warnings.filterwarnings('ignore')
import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
# cancer.head()

##############################
# Voting
# 여러 개의 분류기가 투표를 통해 최종 에측 결과 결정
# Hard Voting: 다수결의 원칙
# Soft Voting: 결정 확률 평균을 구한 뒤 가장 확률이 높은 레이블 값
##############################
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)

# VotingClassfier 활용
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)

V_clf = VotingClassifier(estimators=[('LR', lr), ('KNN', knn)], voting='soft')

V_clf.fit(X_train, y_train)
pred = V_clf.predict(X_test)
print('VotingClassifer 정확도: ', round(accuracy_score(y_test, pred), 4))


models = [lr, knn]
for model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    name = model.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(name, accuracy_score(y_test, pred)))


#########################
# Bagging (Bootstrapping = 반복복원추출)
# 모델을 다양하게 만들기 위해 데이터를 재구성
# 학습 데이터가 충분하지 않더라도 충분한 학습효과
# -> underfitting, overfitting 문제 해결
# 대표적 알고리즘: radom forest
#########################
    
# Random Forest
# 여러 개의 결정 트리 분류기가 전체 데이터에서 bagging 방식으로 각자의 데이터를 샘플링
print("\n")
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=0)
clf.fit(X_train, y_train)
pred1 = clf.predict(X_test)
print('RandomForestClassifier 정확도: ', round(accuracy_score(y_test, pred1), 4))

# Random Forest는 다양한 hyper parameter를 가지기 때문에
# hyper parameter tuning을 해주면 좋음
from sklearn.model_selection import GridSearchCV
params = {
    'n_estimators':[100],
    'max_depth' : [6, 8, 10, 12], 
    'min_samples_leaf' : [8, 12, 18 ],
    'min_samples_split' : [8, 16, 20]
}

rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
grid_cv.fit(X_train, y_train)

print('최적 하이퍼 파라미터\n', grid_cv.best_params_)
print('\n최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))


#########################
# Boosting
# 여러개의 weak learner를 순차적으로 학습 - 예측
# 잘못 예측한 데이터에 가중치 부여해 오류 개선
# high bias를 낮추는 것과 같이 성능 자체를 강화하는데 목적
#########################
# 다양한 알고리즘 중 AdaBoostClassifier와 GradientBoostingClassifier 구현
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=30, random_state=10, learning_rate=0.1)
clf.fit(X_train, y_train)
pred2 = clf.predict(X_test)

print('\n')
print('AdaBoost 정학도: ', round(accuracy_score(y_test, pred2), 4))

from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train, y_train)
pred3 = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, pred3)

print('GBM 정확도: ', round(accuracy_score(y_test, pred3), 4))