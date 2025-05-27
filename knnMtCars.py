# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

# %%
mtCars = pd.read_csv('mt_cars.csv')
mtCars.head()

# %%
X =  mtCars[['mpg','hp']].values
y = mtCars['cyl'].values

# %%
knn = KNeighborsClassifier(n_neighbors=3)
model = knn.fit(X,y)

# %%
model

# %%
y_prev = model.predict(X)
print(y_prev)

# %%
accuracy = accuracy_score(y, y_prev)
precision = precision_score(y, y_prev, average='weighted')
recall = recall_score(y, y_prev, average='weighted')
f1 =f1_score(y, y_prev, average='weighted')
cm = confusion_matrix(y, y_prev)

print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
print("Confusion Matrix:\n", cm)

# %%

                    # mpg , hp
new_data = np.array([[19.3,105]])
previsao = model.predict(new_data)
print(previsao)

distance, indice = model.kneighbors(new_data)
print(distance)
print(indice)

# %%
mtCars.loc[[1,5,31],["cyl","mpg","hp"]]


