import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x = np.array([[1, 2], [2, 4], [3,2]])
y = np.array([0, 0, 1])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)

new_student = np.array([[3,3]])
print(knn.predict(new_student))