import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
# from sklearn.cluster import MeanShift
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, svm, cross_validation

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)
df.drop('Name', 1, inplace=True)
##df_test.drop('Name', 1, inplace=True)
df.convert_objects(convert_numeric=True)
df_test.convert_objects(convert_numeric=True)
pId = df_test['PassengerId']
##print(df_test.head())

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))
##        print(text_digit_vals)    
    return df

df = handle_non_numerical_data(df)
df_test = handle_non_numerical_data(df_test)
##print(df.head())
sns.heatmap(df.corr())
plt.show()

df.drop(['Ticket'], 1, inplace=True)

X = np.array(df.drop(['Survived'],1).astype(float))

X = preprocessing.scale(X)
y = df['Survived']

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

df_test.drop(['Ticket'], 1, inplace=True)

X_test1 = np.array(df_test.drop(['Name'],1).astype(float))
X_test1 = preprocessing.scale(X_test1)


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
xx = clf.predict(X_test1)

print(acc, len(xx), df_test.count(), len(df_test))

##c = 0
with open('submission_file.csv', 'w') as f:
    f.write('PassengerId,Survived\n')

with open('submission_file.csv', 'a') as f:
    for i in range(len(xx)):
        f.write('{},{}\n'.format(pId[i], xx[i]))
        print(pId[i], xx[i])










