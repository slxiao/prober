from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['mc_loc', 'cc','mc_esst_cmpl','mc_dsgn_cmpl',
                    'vocabulary','volume','proglen','difficulty','hls_intl','effort',
                    'hls','timeteqprog','linecount','commentcount','blankcount','linecommentcount',
                    'distoprt','distoper','totoprt','totoper','bc','defects']

CSV_SELECTED_NAMES = ['cc', 'vocabulary','volume','proglen','difficulty','effort',
                    'timeteqprog','linecount','commentcount','blankcount','linecommentcount',
                    'distoprt','distoper','totoprt','totoper','defects']      


def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      column_names = CSV_COLUMN_NAMES,
      batch_size=5, # Artificially small to make examples easier to show.
      select_columns = ['cc', 'vocabulary','volume','proglen','difficulty','effort',
                    'timeteqprog','linecount','commentcount','blankcount','linecommentcount',
                    'distoprt','distoper','totoprt','totoper','defects'],
      label_name='defects',
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
    return dataset

def get_predict_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      select_columns = ['cc', 'vocabulary','volume','proglen','difficulty','effort',
                    'timeteqprog','linecount','commentcount','blankcount','linecommentcount',
                    'distoprt','distoper','totoprt','totoper'],
      batch_size=1, # Artificially small to make examples easier to show.
      na_value="?",
      num_epochs=1,
      shuffle=False,
      ignore_errors=True, 
      **kwargs)
    return dataset

class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels

class PackNumericFeatureWithoutLabels(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features

def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data-mean)/std

def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))

data = get_dataset("data.csv")
data.shuffle(500)

test_dataset = data.take(20) 
train_dataset = data.skip(20)

predict_data = get_predict_dataset("metrics.csv")


# number features preprocessing
NUMERIC_FEATURES = CSV_SELECTED_NAMES[:-1]

packed_train_data = train_dataset.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = test_dataset.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_predict_data = predict_data.map(
    PackNumericFeatureWithoutLabels(NUMERIC_FEATURES))


desc = pd.read_csv("data.csv", header=None, names = CSV_COLUMN_NAMES)[NUMERIC_FEATURES].describe()
normalizer = functools.partial(normalize_numeric_data, mean=np.array(desc.T['mean']), std=np.array(desc.T['std']))
numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]

# model
model = tf.keras.Sequential([
  tf.keras.layers.DenseFeatures(numeric_columns),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])


# train, evaluate and predict
test_data = packed_test_data
model.fit(packed_train_data, epochs=20)

test_loss, test_accuracy = model.evaluate(test_data)

print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))

predictions = model.predict(test_data)

# Show some results
for prediction, defects in zip(predictions[:100], list(test_data)[0][1][:10]):
    print("Predicted defect: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("YES" if bool(defects) else "NO"))

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

def show_batch_without_label(dataset):
  for batch in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))


java_predictions = model.predict(packed_predict_data)

metrics = pd.read_csv("metrics.csv")

results = []
for i in range(len(metrics)):
    results.append((metrics.loc[i,]["filepath"], java_predictions[i][0]))

results.sort(key=lambda x: x[1], reverse=True)

for result in results[0:10]:
    print(result)

show_batch_without_label(packed_predict_data)