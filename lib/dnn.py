import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input, Concatenate, Dropout

def fit_dnn(X_train, y_train, num_epochs, batch_size):
    model = keras.Sequential([
        Input(shape = (8,)),
        Dense(64,activation = 'relu'),
        Dropout(0.2),
        Dense(64,activation = 'relu'),
        Dropout(0.2),
        Dense(64,activation = 'relu'),
        Dropout(0.2),
        Dense(32,activation = 'relu'),
        Dropout(0.2),
        Dense(32,activation = 'relu'),
        Dropout(0.2),
        Dense(16,activation = 'relu'),
        Dropout(0.2),
        Dense(16,activation = 'relu'),
        Dropout(0.2),
        Dense(1,activation = 'sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

    return model