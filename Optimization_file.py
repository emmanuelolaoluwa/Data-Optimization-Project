import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


pkm = pd.read_csv('Pokemon.csv')

pkm1 = pkm.drop(['#','Name','Generation'],1)
pkm_attributes = pkm1.drop(['Legendary','Type 1','Type 2'],1)


from sklearn.decomposition import PCA

#PCA ANALYSIS

pca = PCA(n_components=pkm_attributes.shape[1])
fit = pca.fit(pkm_attributes).transform(pkm_attributes)

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1 = np.insert(var1,0,0)
plt.plot(var1)
axes = plt.gca()
axes.set_ylim([40,110])
plt.show()

pca = PCA(n_components=3)
fit = pca.fit(pkm_attributes).transform(pkm_attributes)
fit1 = pd.DataFrame(fit,columns=['c1','c2','c3'])
df = pd.concat([fit1,pkm1['Legendary']],axis=1)

#------------------------------------------------------------------------------#
#Classification 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

leg = abs((df['Legendary'].values - 1))
X_train, X_test, y_train, y_test = train_test_split(fit, leg, test_size=0.33, random_state=42)


#----------------------------------------------------------------------#

#Decision Tree
from sklearn import tree
from sklearn.model_selection import cross_val_score

# creating odd list
myList = list(range(10,30))

# empty list that will hold cross validation scores
cv_scores = []

# perform 10-fold cross validation we are already familiar with
for k in myList:
    clf = tree.DecisionTreeClassifier(max_depth=k)
    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

#I find the best k with least MSE and visualize it.

# determining best k
optimal_k = myList[MSE.index(min(MSE))]
print("The optimal number of max depth is %d" % optimal_k)
# plot misclassification error vs k
plt.plot(myList, MSE)
plt.xlabel('Number of Max Depth K')
plt.ylabel('Misclassification Error')
plt.show()

#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#


#KNN

from sklearn.neighbors import KNeighborsClassifier
# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cross validation scores
cv_scores = []

# perform 10-fold cross validation we are already familiar with
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

#-----------------------------------------------------------------------------#

from sklearn import svm
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(svm.SVC(), param_grid={"C":[0.001,0.1, 1, 10], "gamma": [1, 0.1, 0.01,0.001]}, cv=4)
grid.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

clf = svm.SVC(C=1,gamma=0.001)
clf.fit(X_train,y_train)
y_predict_svm = clf.predict(X_test)
c_svm = confusion_matrix(y_test, y_predict_svm)
c_svm
