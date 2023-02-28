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


