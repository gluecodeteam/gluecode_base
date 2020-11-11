from config import *
from imports import *

def create_model(modelname, numlabels):
    if modelname == 'DummNN':
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(numlabels, activation='softmax')
        ])
        model.summary()

    if modelname == 'LSTM':
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(numlabels, activation='softmax')
        ])
        model.summary()

    if modelname == 'CNN':
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
            # specify the number of convolutions that you want to learn, their WINDOW size, and their activation function.
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(numlabels, activation='softmax')
        ])
        model.summary()
    return model

