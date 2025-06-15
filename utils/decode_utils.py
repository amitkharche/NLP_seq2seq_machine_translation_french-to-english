import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def decode_sequence_basic(input_seq, encoder_model, decoder_model, tokenizer_out, max_len=20):
    """Decode sequence for basic seq2seq model"""
    # Encode the input
    states_value = encoder_model.predict(input_seq, verbose=0)
    
    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_out.word_index.get('<START>', 1)
    
    # Reverse word index for decoding
    reverse_target_char_index = dict((i, char) for char, i in tokenizer_out.word_index.items())
    
    # Sample loop for one sequence
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index.get(sampled_token_index, '')
        
        if sampled_char == '<END>' or len(decoded_sentence.split()) > max_len:
            stop_condition = True
        else:
            if sampled_char not in ['<START>', '<END>', '']:
                decoded_sentence += sampled_char + ' '
        
        # Update the target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return decoded_sentence.strip()

def decode_sequence_attention(input_seq, model, tokenizer_out, max_len=20):
    """Decode sequence for attention-based model (simplified)"""
    # This is a simplified version - in practice, you'd need inference models
    pred = model.predict([input_seq, input_seq], verbose=0)  # Using input_seq twice as placeholder
    pred_seq = np.argmax(pred[0], axis=-1)
    
    reverse_word_map = dict(map(reversed, tokenizer_out.word_index.items()))
    decoded_sentence = ' '.join([reverse_word_map.get(i, '') for i in pred_seq 
                                if i != 0 and reverse_word_map.get(i, '') not in ['<START>', '<END>', '']])
    
    return decoded_sentence.strip()