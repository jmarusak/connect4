#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from c4.networks.cnn_small import layers
from c4.utils import print_encoded_board

np.random.seed(42)

X = np.load('./data/features.npy')
y = np.load('./data/labels.npy')

print(X.shape)

num_samples = X.shape[0]
num_train_samples = int(0.7 * num_samples)

X_train, X_test = X[:num_train_samples], X[num_train_samples:]
y_train, y_test = y[:num_train_samples], y[num_train_samples:]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


model = Sequential()
for layer in layers((6,7,1)):
    model.add(layer)
model.add(Dense(7, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=10)

# Evaluate the model
score = model.evaluate(X_test, y_test)
print(score[1])

# Predict next move for a sample board
X_true = X_test[0]
y_true = y_test[0]
y_pred_probs = model.predict(np.expand_dims(X_true, axis=0))[0]

print('\nTrained on {} samples'.format(num_samples))
print('\n\nBoard:')
print(X_true.reshape(6, 7))

print('\nTrue Move Labels:')
print(y_true)

print('\nTrue Move:')
print(np.argmax(y_true)+1)

print('\nPred Move Probs:')
print(y_pred_probs)

print('\nPred Move (max):')
print(np.argmax(y_pred_probs)+1)

print('\nPred Move (sampling dist):')
print(np.random.choice([1,2,3,4,5,6,7], 1, p=y_pred_probs)+1)


print_encoded_board(X_true)
