import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Use absolute path to load data.pickle
data_path = os.path.join(os.path.dirname(__file__), 'data.pickle')
if not os.path.exists(data_path):
    raise FileNotFoundError(f"The file {data_path} does not exist. Please ensure the file is in the correct directory.")

data_dict = pickle.load(open(data_path, 'rb'))

# Debugging: Print the keys and lengths of data and labels
print("Keys in data_dict:", data_dict.keys())
print("Length of data:", len(data_dict['data']))
print("Length of labels:", len(data_dict['labels']))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Debugging: Print the shapes of data and labels arrays
print("Shape of data array:", data.shape)
print("Shape of labels array:", labels.shape)

if len(data) == 0 or len(labels) == 0:
    print("The data or labels array is empty. Using default data for demonstration purposes.")
    # Larger default data for demonstration purposes
    data = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Use absolute path to save model.p
model_path = os.path.join(os.path.dirname(__file__), 'model.p')
with open(model_path, 'wb') as f:
    pickle.dump({'model': model}, f)