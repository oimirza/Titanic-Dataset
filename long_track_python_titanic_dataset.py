import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('titanic.csv')

print(df.dtypes)

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Age'] = df['Age'].astype(int)

print(df.dtypes)

print(df.isna().sum())

df.drop('Cabin', axis=1, inplace=True)

df.drop('PassengerId', axis=1, inplace=True)
df.drop('Name', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)
df.drop('Fare', axis=1, inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print(df.isna().sum())

df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})

df['Embarked'] = df['Embarked'].replace({'S': 0, 'C': 1,'Q': 2})

df[df.duplicated()]

print(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Sex', data=df)
plt.title('Number of Passengers by Sex')
plt.ylabel('Number of Passengers')
plt.show()

df.groupby('Survived').size().plot(kind = 'pie', 
                                   autopct = '%.2f%%', 
                                   labels = ['Died', 'Survived'], 
                                   label = '', 
                                   fontsize = 10,
                                   colors = ['red', 'green']);

sns.histplot(x='Age', data=df, kde=True)
plt.title('Number of passengers group by Ages')
plt.ylabel('Number of Passengers')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Number of Survivors vs Non-Survivors by Class')
plt.xlabel('Passenger by Class')
plt.ylabel('Number of Passengers')
plt.show()

sns.countplot(x='Sex', data=df, hue='Survived')
plt.title('Number of Survivors vs Non-Survivors by Gender')
plt.ylabel('Number of Passengers')
plt.show()

!pip install sweetviz

import sweetviz as sv

advert_report = sv.analyze(df)

advert_report.show_html('titanic.html')
advert_report.show_notebook()

from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)
y = df['Survived']

X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
lm1= LogisticRegression()
lm1.fit(X_train, y_train)


ypred=lm1.predict(X_test)
ypred


from sklearn.metrics import accuracy_score
accuracy_score(y_test, ypred)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, ypred)
print(confusion_matrix)
