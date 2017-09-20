import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

sns.set_style('whitegrid')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# data clean
full_data = [train, test]


train['HasCabin'] = train['Cabin'].notnull().astype(int)

for dataset in full_data:
    dataset['FamilySize'] = train['SibSp'] + train['Parch'] + 1

for dataset in full_data:
    dataset['IsAlone'] = (dataset['FamilySize'] == 1).astype(int)

for dataset in full_data:
    dataset['Fare']=dataset['Fare'].fillna(dataset['Fare'].median())

for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map({"female":0, "male":1})

for dataset in full_data:
    age_mean = dataset['Age'].mean()
    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_mean-age_std,age_mean+age_std,size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64), 'Age'] = 4

    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

drop_elems = ['PassengerId', 'Name', 'Embarked', 'SibSp', 'Parch', 'Ticket','Cabin']

train = train.drop(drop_elems, axis=1)
test = test.drop(drop_elems, axis=1)

print(train['Sex'].isnull().sum())
print(train['Age'].isnull().sum())
print("female sum: %d" % (train['Sex'] == 0).sum())
print("male sum: %d" % (train['Sex'] == 1).sum())
print(train[train['Survived']==0].count())

isAloneCnt = train.loc[train['IsAlone'] == 1, 'Survived'].count()
isAloneAndSurvived = train.loc[(train['IsAlone'] == 1) & (train['Survived'] == 1), 'Survived'].count()
print(isAloneAndSurvived.astype(float)/isAloneCnt)

isNonAloneCnt = train.loc[train['IsAlone'] == 0, 'Survived'].count()
isNonAloneAndSurvived = train.loc[(train['IsAlone'] == 0) & (train['Survived'] == 1), 'Survived'].count()
print(isNonAloneAndSurvived.astype(float)/isNonAloneCnt)

print(train.loc[(train['IsAlone'] == 0), 'Survived'].count())

sns.barplot(x='Pclass', y='Survived', data=train)
plt.show()
sns.barplot(x='Sex', y='Survived', data=train)
plt.show()
sns.barplot(x='Age', y='Survived', data=train,ci=0)
plt.show()
sns.barplot(x='Fare', y='Survived', data=train,ci=0)
plt.show()
sns.barplot(x='HasCabin', y='Survived', data=train,ci=0)
plt.show()
sns.barplot(x='IsAlone', y='Survived',data=train)
plt.show()

colormap = plt.cm.viridis
plt.title('Pearson Correlation of Features')
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()


