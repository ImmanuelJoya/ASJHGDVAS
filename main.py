import sklearn  # for machine learning algorithms
import numpy as np  # for numerical operations
import matplotlib.pyplot as plt  # for plotting and visualization
import matplotlib  # for version info
import pandas as pd  # for data manipulation and analysis
import tensorflow as tf  # for deep learning and neural networks

# Load dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dftest = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# Remove the 'survived' column (lab   els)
y_train = dftrain.pop('survived')
y_eval = dftest.pop('survived')

# Feature columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(
        tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
    )

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

# Version info
print("version information:-")
print("TensorFlow version:", tf.__version__)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Matplotlib version:", matplotlib.__version__)
