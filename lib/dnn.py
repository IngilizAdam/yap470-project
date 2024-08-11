import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input, Concatenate, Dropout, BatchNormalization

def fit_basic_dnn(X_train, y_train, num_epochs, batch_size):
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
        BatchNormalization(),
        Dense(32,activation = 'relu'),
        Dropout(0.2),
        BatchNormalization(),
        Dense(16,activation = 'relu'),
        Dropout(0.2),
        BatchNormalization(),
        Dense(16,activation = 'relu'),
        Dropout(0.2),
        BatchNormalization(),
        Dense(1,activation = 'sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

    return model

def fit_embedded_dnn(X_train, y_train, num_epochs, batch_size):
    # Define the embedding dimension
    embedding_dim = 10

    unique_carrier_dim = len(X_train['UniqueCarrier'].unique())
    origin_dim = len(X_train['Origin'].unique())
    dest_dim = len(X_train['Dest'].unique())

    # Embedding layers for categorical columns
    input_1 = Input(shape=[1])
    input_2 = Input(shape=[1])
    input_3 = Input(shape=[1])
    input_4 = Input(shape=[1])
    input_5 = Input(shape=[1])
    input_6 = Input(shape=[1])
    input_7 = Input(shape=[1])
    input_8 = Input(shape=[1])

    embedding_1 = Embedding(input_dim=unique_carrier_dim, output_dim=embedding_dim)(input_1)
    embedding_2 = Embedding(input_dim=origin_dim, output_dim=embedding_dim)(input_2)
    embedding_3 = Embedding(input_dim=dest_dim, output_dim=embedding_dim)(input_3)

    # Flatten the embeddings
    flatten_1 = Flatten()(embedding_1)
    flatten_2 = Flatten()(embedding_2)
    flatten_3 = Flatten()(embedding_3)

    # Concatenate all input layers
    concatenated = Concatenate()([flatten_1, flatten_2, flatten_3, input_4, input_5, input_6, input_7, input_8])

    # Fully connected layers
    dense_1 = Dense(64, activation='relu')(concatenated)
    dropout_1 = Dropout(0.2)(dense_1)
    dense_2 = Dense(64, activation='relu')(dropout_1)
    dropout_2 = Dropout(0.2)(dense_2)
    dense_3 = Dense(64, activation='relu')(dropout_2)
    dropout_3 = Dropout(0.2)(dense_3)
    dense_4 = Dense(32, activation='relu')(dropout_3)
    dropout_4 = Dropout(0.2)(dense_4)
    batch_norm_1 = BatchNormalization()(dropout_4)
    dense_5 = Dense(32, activation='relu')(batch_norm_1)
    dropout_5 = Dropout(0.2)(dense_5)
    batch_norm_2 = BatchNormalization()(dropout_5)
    dense_6 = Dense(16, activation='relu')(batch_norm_2)
    dropout_6 = Dropout(0.2)(dense_6)
    batch_norm_3 = BatchNormalization()(dropout_6)
    dense_7 = Dense(16, activation='relu')(batch_norm_3)
    dropout_7 = Dropout(0.2)(dense_7)

    # Output layer
    output = Dense(1, activation='sigmoid')(dropout_7)

    # Create the model
    model = Model(inputs=[input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8], outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit([X_train['UniqueCarrier'], X_train['Origin'], X_train['Dest'], X_train['Month'], X_train['DayofMonth'], X_train['DayOfWeek'], X_train['DepTime'], X_train['Distance']], y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
    
    return model