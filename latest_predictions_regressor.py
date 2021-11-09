import numpy as np
from sklearn.ensemble import RandomForestClassifier

print(np.__file__)

myInput = np.array([[1, 2], [2, 3], [4, 5], [1, 8]])

target = np.array([0, 1, 0, 0])

clf = RandomForestClassifier(n_estimators=3)

clf.fit(myInput, target)

clf.predict(myInput)

for e in clf.estimators_:
    print(e.latest_predictions)