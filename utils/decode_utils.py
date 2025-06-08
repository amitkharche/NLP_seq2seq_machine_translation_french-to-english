import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def decode_sequence(input_seq, model, tokenizer_out, max_len=10):
    pred = model.predict(input_seq)
    pred_seq = np.argmax(pred[0], axis=-1)
    reverse_word_map = dict(map(reversed, tokenizer_out.word_index.items()))
    decoded_sentence = ' '.join([reverse_word_map.get(i, '') for i in pred_seq if i != 0])
    return decoded_sentence.strip()
