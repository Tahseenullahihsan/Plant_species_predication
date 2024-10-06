import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
data=sns.load_dataset('iris')
X=data.drop(columns=['species'])

Y=data['species']
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train )
model.predict([[5.1,3.5,1.4,0.2]])
## sepal_length,sepal_width,petal_length,petal_width

sns.scatterplot(x='sepal_length', y='sepal_width',
                hue='species', data=data, )

# Placing Legend outside the Figure
data.legend(bbox_to_anchor=(1, 1), loc=2)

plt.show()