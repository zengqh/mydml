import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV

data = pd.read_csv('diabetes.csv')
count_classes = pd.value_counts(data['Outcome'], sort = True).sort_index()
count_classes.plot(kind='bar')
plt.title('outcome histogram')
plt.xlabel("Conversion")
plt.ylabel("Count")
plt.show()

y = data.Outcome
X = data.drop('Outcome', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"
        ]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

results={}
for name, clf in zip(names, classifiers):
    scores = cross_val_score(clf, X_train, y_train, cv = 5)
    results[name] = scores

for name, scores in results.items():
    print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100 * scores.mean(), 100 * scores.std() * 2))

clf = SVC(kernel="linear")
param_grid = [
    {'C':[0.01,0.1,1,10], 'kernel':['linear']},
]

grid = GridSearchCV(estimator=clf, param_grid=param_grid)
grid.fit(X_train, y_train)
print(grid)

print("Best score: %0.2f%%" % (100*grid.best_score_))
print("Best estimator for parameter C: %f" % (grid.best_estimator_.C))

clf = SVC(kernel='linear', C=0.1)
clf.fit(X_train, y_train)
y_eval = clf.predict(X_test)

acc = sum(y_eval == y_test) / float(len(y_test))
print("Accuracy: %.2f%%" % (100*acc))