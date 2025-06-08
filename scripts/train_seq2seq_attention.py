import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
import numpy as np

# Dummy data
def load_data():
    input_texts = ["bonjour", "salut", "merci"]
    target_texts = ["hello", "hi", "thanks"]
    return input_texts, target_texts

def preprocess_texts(input_texts, target_texts, max_len=10):
    tokenizer_in = tf.keras.preprocessing.text.Tokenizer()
    tokenizer_out = tf.keras.preprocessing.text.Tokenizer()
    tokenizer_in.fit_on_texts(input_texts)
    tokenizer_out.fit_on_texts(target_texts)

    encoder_input_data = tokenizer_in.texts_to_sequences(input_texts)
    decoder_input_data = tokenizer_out.texts_to_sequences(target_texts)

    encoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_data, maxlen=max_len)
    decoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_data, maxlen=max_len)

    return encoder_input_data, decoder_input_data, tokenizer_in, tokenizer_out

input_texts, target_texts = load_data()
encoder_input_data, decoder_input_data, tokenizer_in, tokenizer_out = preprocess_texts(input_texts, target_texts)

latent_dim = 256
vocab_in = len(tokenizer_in.word_index) + 1
vocab_out = len(tokenizer_out.word_index) + 1
max_len = encoder_input_data.shape[1]

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_in, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(vocab_out, latent_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# Attention Mechanism
score_dense = Dense(1, activation="tanh")
attention_scores = score_dense(encoder_outputs)
attention_weights = tf.nn.softmax(attention_scores, axis=1)
context_vector = tf.reduce_sum(attention_weights * encoder_outputs, axis=1)
context_vector = tf.expand_dims(context_vector, 1)
decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_outputs])

# Final output
output = Dense(vocab_out, activation="softmax")(decoder_combined_context)

model = Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
model.save("../models/attention_seq2seq.h5")
