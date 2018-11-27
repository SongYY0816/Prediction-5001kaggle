import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

##------read csv file (train and test)
train = pd.read_csv("/Users/song/Downloads/trainnew.csv")
test = pd.read_csv("/Users/song/Downloads/test.csv")

#train.head(10)
#test.head(10)

##------divide features into categorical and numerical
##------convert categorical variables into numeric by using labelEncoder
le = preprocessing.LabelEncoder()
cat_vars1=train['penalty']
cat_vars2=test['penalty']
train['penalty'] = le.fit_transform(cat_vars1.tolist())
test['penalty'] = le.fit_transform(cat_vars2.tolist())

##------feature engineering
'''
X = np.array(X[['penalty','l1_ratio','alpha','max_iter','random_state', 'n_jobs',
                   'n_samples','n_features','n_classes','n_clusters_per_class',
                   'n_informative','flip_y','scale']])
Y = np.array(train['time'])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)
#find importance
feat_labels = train.columns[1:]
forest = RandomForestRegressor(n_estimators = 10000, random_state = 0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
'''
##multiply some related features
train['other']=train['max_iter']*train['n_samples']*train['n_features']
test['other']=test['max_iter']*test['n_samples']*test['n_features']
##square features which impotances are high
train['scale']=np.square(train['scale'])
test['scale']=np.square(test['scale'])
#train['max_iter']=np.square(train['max_iter'])
#test['max_iter']=np.square(test['max_iter'])
#train['n_features']=np.square(train['n_features'])
#test['n_features']=np.square(test['n_features'])
train['n_classes']=np.square(train['n_classes'])
test['n_classes']=np.square(test['n_classes'])
##replace the nagtive value with positive one
##the number of n_jobs also can be given as exponent of 2
train['n_jobs']=train['n_jobs'].replace([-1],[16])
test['n_jobs']=test['n_jobs'].replace([-1],[16])


##------model training
X = train[['other','n_jobs','n_classes',
           'n_clusters_per_class','n_informative','flip_y','scale','penalty']]

Y = train['time']

Z = test[['other', 'n_jobs','n_classes','n_clusters_per_class',
                   'n_informative','flip_y','scale','penalty']]


##------adjust parameters
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
for i in range(0,50):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=i)
    random.seed(1000)
    rf = RandomForestRegressor(max_depth=10,n_estimators=200)
    rf.fit(X_train, y_train)
    m=rf.predict(X_test)
    n=y_test
    print(i,metrics.mean_squared_error(n,m))

for i in range(2,70):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)
    random.seed(1000)
    rf = RandomForestRegressor(max_depth=i,n_estimators=500)
    rf.fit(X_train, y_train)
    m=rf.predict(X_test)
    n=y_test
    print(i,metrics.mean_squared_error(n,m))

p=[100,200,300,400,500,600]
for i in p:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)
    random.seed(1000)
    rf = RandomForestRegressor(max_depth=19,n_estimators=i)
    rf.fit(X_train, y_train)
    m=rf.predict(X_test)
    n=y_test
    print(metrics.mean_squared_error(n,m))

p=[0.2,0.25,0.3,0.35,0.4]
for i in p:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=i, random_state=7)
    random.seed(1000)
    rf = RandomForestRegressor(max_depth=19,n_estimators=400)
    rf.fit(X_train, y_train)
    m=rf.predict(X_test)
    n=y_test
    print(metrics.mean_squared_error(n,m))
'''

'''
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=1)

depths = [1,10,20,30,40,50]
errors1 = []

for depth in depths:
    forest = RandomForestRegressor(max_depth=depth)
    forest.fit(X_train, y_train)
  
    z1 = forest.predict(X_test)
    error1 = np.sqrt(np.sum((y_test - z1) * (y_test - z1)) / (1.0 * len(y_test)))
    errors1.append(error1)
print(errors1)
'''

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
random.seed(1000)
rf = RandomForestRegressor(max_depth=11,n_estimators=600)
rf.fit(X_train, y_train)

m=rf.predict(X_test)
n=y_test
#metrics.mean_squared_error(n,m)

##------prediction
final_status = rf.predict(Z)
final_status=pd.DataFrame(final_status, columns=['time'])
final_status.index.name='id'
final_status.to_csv('test.csv')