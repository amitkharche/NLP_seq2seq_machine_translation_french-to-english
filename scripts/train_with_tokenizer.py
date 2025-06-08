import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.tokenizer_utils import create_tokenizers, save_tokenizers

# Load sample dataset
lines = open("data/french_english_pairs.txt", encoding='utf-8').read().strip().split('\n')
input_texts, target_texts = zip(*[line.split('\t') for line in lines])

# Tokenize
tokenizer_in, tokenizer_out = create_tokenizers(input_texts, target_texts)
input_seq = tokenizer_in.texts_to_sequences(input_texts)
target_seq = tokenizer_out.texts_to_sequences(target_texts)

max_len = 10
encoder_input_data = pad_sequences(input_seq, maxlen=max_len, padding='post')
decoder_input_data = pad_sequences(target_seq, maxlen=max_len, padding='post')

# Save tokenizers
save_tokenizers(tokenizer_in, tokenizer_out, "utils/tokenizer_in.pkl", "utils/tokenizer_out.pkl")

# Define simple seq2seq model
latent_dim = 128
vocab_in = len(tokenizer_in.word_index) + 1
vocab_out = len(tokenizer_out.word_index) + 1

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_in, latent_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_out, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_out, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Compile and train
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], 
          tf.expand_dims(decoder_input_data, -1), 
          batch_size=2, epochs=30, verbose=1)

model.save("models/basic_seq2seq.h5")
