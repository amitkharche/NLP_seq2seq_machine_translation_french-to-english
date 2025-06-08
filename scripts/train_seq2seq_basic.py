import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np
import os

# Dummy data loader for illustration
def load_data():
    input_texts = ["bonjour", "salut", "merci"]
    target_texts = ["hello", "hi", "thanks"]
    return input_texts, target_texts

def preprocess_texts(input_texts, target_texts, num_samples=10000, max_len=10):
    tokenizer_in = tf.keras.preprocessing.text.Tokenizer()
    tokenizer_out = tf.keras.preprocessing.text.Tokenizer()
    tokenizer_in.fit_on_texts(input_texts)
    tokenizer_out.fit_on_texts(target_texts)

    encoder_input_data = tokenizer_in.texts_to_sequences(input_texts)
    decoder_input_data = tokenizer_out.texts_to_sequences(target_texts)

    encoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_data, maxlen=max_len)
    decoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_data, maxlen=max_len)

    return encoder_input_data, decoder_input_data, tokenizer_in, tokenizer_out

# Load and preprocess data
input_texts, target_texts = load_data()
encoder_input_data, decoder_input_data, tokenizer_in, tokenizer_out = preprocess_texts(input_texts, target_texts)

# Define model
latent_dim = 256
vocab_in = len(tokenizer_in.word_index) + 1
vocab_out = len(tokenizer_out.word_index) + 1
max_len = encoder_input_data.shape[1]

encoder_inputs = Input(shape=(None,))
x = Embedding(vocab_in, latent_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
y = Embedding(vocab_out, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(y, initial_state=encoder_states)
decoder_dense = Dense(vocab_out, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
model.save("../models/basic_seq2seq.h5")
