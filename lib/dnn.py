import tensorflow as tf
from tensorflow import keras

def fit_dnn(X_train, y_train, num_epochs, batch_size):
    # Initialize the model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model to the training data
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)

    return model