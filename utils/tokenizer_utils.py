import tensorflow as tf
import pickle
import os

def create_tokenizers(input_texts, target_texts):
    """Create and fit tokenizers for input and target languages"""
    tokenizer_in = tf.keras.preprocessing.text.Tokenizer(filters='', lower=True)
    tokenizer_out = tf.keras.preprocessing.text.Tokenizer(filters='', lower=True)
    
    # Add special tokens
    input_texts = ['<START> ' + text + ' <END>' for text in input_texts]
    target_texts = ['<START> ' + text + ' <END>' for text in target_texts]
    
    tokenizer_in.fit_on_texts(input_texts)
    tokenizer_out.fit_on_texts(target_texts)
    
    return tokenizer_in, tokenizer_out

def save_tokenizers(tokenizer_in, tokenizer_out, path_in='tokenizer_in.pkl', path_out='tokenizer_out.pkl'):
    """Save tokenizers to pickle files"""
    os.makedirs(os.path.dirname(path_in), exist_ok=True)
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    
    with open(path_in, 'wb') as f:
        pickle.dump(tokenizer_in, f)
    with open(path_out, 'wb') as f:
        pickle.dump(tokenizer_out, f)

def load_tokenizers(path_in='tokenizer_in.pkl', path_out='tokenizer_out.pkl'):
    """Load tokenizers from pickle files"""
    with open(path_in, 'rb') as f:
        tokenizer_in = pickle.load(f)
    with open(path_out, 'rb') as f:
        tokenizer_out = pickle.load(f)
    return tokenizer_in, tokenizer_out