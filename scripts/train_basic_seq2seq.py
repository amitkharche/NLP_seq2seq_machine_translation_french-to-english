import os
import sys

# ✅ Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 🔍 Debug check
print("🧭 Using project root:", project_root)


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import sys
sys.path.append('..')
from utils.tokenizer_utils import create_tokenizers, save_tokenizers

def load_data(filepath="data/french_english_pairs.txt"):
    """Load French-English pairs from file"""
    lines = open(filepath, encoding='utf-8').read().strip().split('\n')
    pairs = [line.split('\t') for line in lines if '\t' in line]
    input_texts, target_texts = zip(*pairs)
    return list(input_texts), list(target_texts)

def prepare_data(input_texts, target_texts, max_len=20):
    """Prepare and tokenize data"""
    # Create tokenizers
    tokenizer_in, tokenizer_out = create_tokenizers(input_texts, target_texts)
    
    # Add special tokens
    input_texts_processed = ['<START> ' + text + ' <END>' for text in input_texts]
    target_texts_processed = ['<START> ' + text + ' <END>' for text in target_texts]
    
    # Convert to sequences
    input_seq = tokenizer_in.texts_to_sequences(input_texts_processed)
    target_seq = tokenizer_out.texts_to_sequences(target_texts_processed)
    
    # Pad sequences
    encoder_input_data = pad_sequences(input_seq, maxlen=max_len, padding='post')
    decoder_input_data = pad_sequences(target_seq, maxlen=max_len, padding='post')
    
    # Decoder target data (shifted by one position)
    decoder_target_data = np.zeros_like(decoder_input_data)
    decoder_target_data[:, :-1] = decoder_input_data[:, 1:]
    
    return encoder_input_data, decoder_input_data, decoder_target_data, tokenizer_in, tokenizer_out

def build_basic_seq2seq(vocab_in, vocab_out, latent_dim=256):
    """Build basic seq2seq model"""
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    enc_emb = Embedding(vocab_in, latent_dim, mask_zero=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True, name='encoder_lstm')(enc_emb)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    dec_emb = Embedding(vocab_out, latent_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(vocab_out, activation='softmax', name='decoder_output')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Training model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # Inference models
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    dec_emb2 = dec_emb
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)
    
    return model, encoder_model, decoder_model

if __name__ == "__main__":
    import numpy as np
    
    # Load and prepare data
    input_texts, target_texts = load_data()
    encoder_input_data, decoder_input_data, decoder_target_data, tokenizer_in, tokenizer_out = prepare_data(input_texts, target_texts)
    
    # Model parameters
    vocab_in = len(tokenizer_in.word_index) + 1
    vocab_out = len(tokenizer_out.word_index) + 1
    
    print(f"Vocabulary sizes - Input: {vocab_in}, Output: {vocab_out}")
    print(f"Data shapes - Encoder: {encoder_input_data.shape}, Decoder: {decoder_input_data.shape}")
    
    # Build model
    model, encoder_model, decoder_model = build_basic_seq2seq(vocab_in, vocab_out)
    
    # Compile
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('models/basic_seq2seq_best.h5', save_best_only=True)
    ]
    
    # Train
    history = model.fit([encoder_input_data, decoder_input_data],
                       np.expand_dims(decoder_target_data, -1),
                       batch_size=32,
                       epochs=100,
                       validation_split=0.2,
                       callbacks=callbacks,
                       verbose=1)
    
    # Save models
    model.save("models/basic_seq2seq.h5")
    encoder_model.save("models/basic_encoder.h5")
    decoder_model.save("models/basic_decoder.h5")
    
    # Save tokenizers
    save_tokenizers(tokenizer_in, tokenizer_out, "models/tokenizer_in.pkl", "models/tokenizer_out.pkl")
    
    print("Training completed and models saved!")
