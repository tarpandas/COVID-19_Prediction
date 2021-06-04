import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import pickle as pkl

df = pd.read_csv('CovidDataset.csv')

del df['Wearing Masks']
del df['Sanitization from Market']

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

encoding = OrdinalEncoder()
x = encoding.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1)

os = RandomOverSampler()
x_train_ns,y_train_ns = os.fit_resample(x_train,y_train)

values_y, count_y = np.unique(y_train_ns, return_counts = True)
print(dict(zip(values_y, count_y)))

result = DecisionTreeClassifier()
result.fit(x_train_ns, y_train_ns)

model = result
result1 = result.predict(x_test)

prediction = accuracy_score(y_test, result1)
print("Accuracy =", prediction * 100, "%")

pkl.dump(model, open('CovidDataset.pkl', 'wb'))