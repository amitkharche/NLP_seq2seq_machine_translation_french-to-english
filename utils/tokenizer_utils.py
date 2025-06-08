import tensorflow as tf
import pickle

def create_tokenizers(input_texts, target_texts):
    tokenizer_in = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer_out = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer_in.fit_on_texts(input_texts)
    tokenizer_out.fit_on_texts(target_texts)
    return tokenizer_in, tokenizer_out

def save_tokenizers(tokenizer_in, tokenizer_out, path_in='tokenizer_in.pkl', path_out='tokenizer_out.pkl'):
    with open(path_in, 'wb') as f:
        pickle.dump(tokenizer_in, f)
    with open(path_out, 'wb') as f:
        pickle.dump(tokenizer_out, f)

def load_tokenizers(path_in='tokenizer_in.pkl', path_out='tokenizer_out.pkl'):
    with open(path_in, 'rb') as f:
        tokenizer_in = pickle.load(f)
    with open(path_out, 'rb') as f:
        tokenizer_out = pickle.load(f)
    return tokenizer_in, tokenizer_out
