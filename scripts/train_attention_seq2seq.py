import os
import sys

# ✅ Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 🔍 Debug check
print("🧭 Using project root:", project_root)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Lambda, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import sys
sys.path.append('..')
from utils.tokenizer_utils import create_tokenizers, save_tokenizers

class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention layer with proper shape handling"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape should be [(batch, seq_len, features), (batch, seq_len, features)]
        encoder_shape, decoder_shape = input_shape
        
        self.W_a = self.add_weight(
            name='W_a',
            shape=(encoder_shape[-1], encoder_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.U_a = self.add_weight(
            name='U_a',
            shape=(decoder_shape[-1], encoder_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.V_a = self.add_weight(
            name='V_a',
            shape=(encoder_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        encoder_outputs, decoder_outputs = inputs
        
        # Get the last decoder output for attention computation
        decoder_last = decoder_outputs[:, -1, :]  # (batch, features)
        
        # Calculate attention scores
        # encoder_outputs: (batch, seq_len, features)
        # decoder_last: (batch, features)
        
        # Expand decoder_last to match encoder sequence length
        decoder_expanded = tf.expand_dims(decoder_last, 1)  # (batch, 1, features)
        decoder_tiled = tf.tile(decoder_expanded, [1, tf.shape(encoder_outputs)[1], 1])
        
        # Compute attention
        encoder_W = tf.tensordot(encoder_outputs, self.W_a, axes=[[2], [0]])
        decoder_U = tf.tensordot(decoder_tiled, self.U_a, axes=[[2], [0]])
        
        tanh_output = tf.tanh(encoder_W + decoder_U)
        attention_scores = tf.tensordot(tanh_output, self.V_a, axes=[[2], [0]])
        attention_scores = tf.squeeze(attention_scores, axis=-1)
        
        # Apply softmax
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        
        # Compute context vector
        context = tf.reduce_sum(encoder_outputs * attention_weights, axis=1)
        
        return context, attention_weights
    
    def compute_output_shape(self, input_shape):
        encoder_shape, decoder_shape = input_shape
        return [(encoder_shape[0], encoder_shape[2]), (encoder_shape[0], encoder_shape[1], 1)]

def build_attention_seq2seq(vocab_in, vocab_out, latent_dim=256, max_len=20):
    """Build attention-based seq2seq model with proper shape handling"""
    
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    enc_emb = Embedding(vocab_in, latent_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    
    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    dec_emb = Embedding(vocab_out, latent_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    
    # Custom attention mechanism
    attention_layer = AttentionLayer()
    context, attention_weights = attention_layer([encoder_outputs, decoder_outputs])
    
    # Expand context to match decoder sequence length
    context_expanded = Lambda(lambda x: tf.expand_dims(x[0], 1))(context)
    context_tiled = Lambda(lambda x: tf.tile(x, [1, tf.shape(decoder_outputs)[1], 1]))(context_expanded)
    
    # Combine context and decoder output
    decoder_combined_context = Concatenate(axis=-1)([context_tiled, decoder_outputs])
    
    # Output layer
    output = Dense(vocab_out, activation='softmax', name='output_layer')(decoder_combined_context)
    
    model = Model([encoder_inputs, decoder_inputs], output)
    
    return model

def build_simple_attention_seq2seq(vocab_in, vocab_out, latent_dim=256, max_len=20):
    """Alternative simpler attention implementation"""
    
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    enc_emb = Embedding(vocab_in, latent_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    
    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    dec_emb = Embedding(vocab_out, latent_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    
    # Simplified attention using dense layers
    attention_dense = Dense(latent_dim, activation='tanh', name='attention_dense')
    attention_v = Dense(1, name='attention_v')
    
    def attention_mechanism(inputs):
        encoder_out, decoder_out = inputs
        
        # Get last decoder state
        decoder_last = decoder_out[:, -1:, :]  # Keep dimension
        
        # Calculate attention scores
        decoder_repeated = tf.tile(decoder_last, [1, tf.shape(encoder_out)[1], 1])
        combined = tf.concat([encoder_out, decoder_repeated], axis=-1)
        
        attention_hidden = attention_dense(combined)
        attention_scores = attention_v(attention_hidden)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # Apply attention
        context = tf.reduce_sum(encoder_out * attention_weights, axis=1, keepdims=True)
        context_repeated = tf.tile(context, [1, tf.shape(decoder_out)[1], 1])
        
        return tf.concat([decoder_out, context_repeated], axis=-1)
    
    # Apply attention
    combined_output = Lambda(attention_mechanism)([encoder_outputs, decoder_outputs])
    
    # Output layer
    output = Dense(vocab_out, activation='softmax', name='output_layer')(combined_output)
    
    model = Model([encoder_inputs, decoder_inputs], output)
    
    return model

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

if __name__ == "__main__":
    # Load and prepare data
    input_texts, target_texts = load_data()
    encoder_input_data, decoder_input_data, decoder_target_data, tokenizer_in, tokenizer_out = prepare_data(input_texts, target_texts)
    
    # Model parameters
    vocab_in = len(tokenizer_in.word_index) + 1
    vocab_out = len(tokenizer_out.word_index) + 1
    
    print(f"Vocabulary sizes - Input: {vocab_in}, Output: {vocab_out}")
    print(f"Data shapes - Encoder: {encoder_input_data.shape}, Decoder: {decoder_input_data.shape}")
    
    # Build model - try the simpler version first
    try:
        print("Building attention model...")
        model = build_simple_attention_seq2seq(vocab_in, vocab_out)
        print("✅ Simple attention model built successfully")
    except Exception as e:
        print(f"❌ Simple attention failed: {e}")
        print("Trying custom attention layer...")
        try:
            model = build_attention_seq2seq(vocab_in, vocab_out)
            print("✅ Custom attention model built successfully")
        except Exception as e2:
            print(f"❌ Custom attention also failed: {e2}")
            print("Falling back to basic seq2seq without attention...")
            # Fallback to basic seq2seq
            encoder_inputs = Input(shape=(None,))
            enc_emb = Embedding(vocab_in, 256, mask_zero=True)(encoder_inputs)
            encoder_lstm = LSTM(256, return_state=True)
            _, state_h, state_c = encoder_lstm(enc_emb)
            
            decoder_inputs = Input(shape=(None,))
            dec_emb = Embedding(vocab_out, 256, mask_zero=True)(decoder_inputs)
            decoder_lstm = LSTM(256, return_sequences=True)
            decoder_outputs = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
            output = Dense(vocab_out, activation='softmax')(decoder_outputs)
            
            model = Model([encoder_inputs, decoder_inputs], output)
            print("✅ Basic seq2seq model built successfully")
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Compile
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('models/attention_seq2seq_best.h5', save_best_only=True)
    ]
    
    # Train
    print("\nStarting training...")
    history = model.fit([encoder_input_data, decoder_input_data],
                       np.expand_dims(decoder_target_data, -1),
                       batch_size=32,
                       epochs=100,
                       validation_split=0.2,
                       callbacks=callbacks,
                       verbose=1)
    
    # Save model
    model.save("models/attention_seq2seq.h5")
    
    # Save tokenizers
    save_tokenizers(tokenizer_in, tokenizer_out, "models/tokenizer_in.pkl", "models/tokenizer_out.pkl")
    
    print("Training completed and models saved!")