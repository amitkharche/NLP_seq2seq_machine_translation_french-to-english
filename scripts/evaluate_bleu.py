import os
import sys

# ✅ Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 🔍 Debug check
print("🧭 Using project root:", project_root)


import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')
from utils.tokenizer_utils import load_tokenizers
from utils.decode_utils import decode_sequence_basic, decode_sequence_attention
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required NLTK data
nltk.download('punkt', quiet=True)

def evaluate_model(model_type="basic"):
    """Evaluate model using BLEU score"""
    
    # Load test data (using same data for simplicity)
    lines = open("../data/french_english_pairs.txt", encoding='utf-8').read().strip().split('\n')
    test_pairs = [line.split('\t') for line in lines[:5] if '\t' in line]  # First 5 for testing
    input_texts, target_texts = zip(*test_pairs)
    
    # Load tokenizers
    tokenizer_in, tokenizer_out = load_tokenizers("../models/tokenizer_in.pkl", "../models/tokenizer_out.pkl")
    
    # Prepare input sequences
    input_texts_processed = ['<START> ' + text + ' <END>' for text in input_texts]
    input_seqs = tokenizer_in.texts_to_sequences(input_texts_processed)
    input_seqs_padded = pad_sequences(input_seqs, maxlen=20, padding='post')
    
    # Load model and generate predictions
    if model_type == "basic":
        encoder_model = load_model("../models/basic_encoder.h5")
        decoder_model = load_model("../models/basic_decoder.h5")
        predictions = []
        for seq in input_seqs_padded:
            pred = decode_sequence_basic(seq.reshape(1, -1), encoder_model, decoder_model, tokenizer_out)
            predictions.append(pred)
    else:
        model = load_model("../models/attention_seq2seq.h5")
        predictions = []
        for seq in input_seqs_padded:
            pred = decode_sequence_attention(seq.reshape(1, -1), model, tokenizer_out)
            predictions.append(pred)
    
    # Calculate BLEU scores
    smoothing_fn = SmoothingFunction().method1
    bleu_scores = []
    
    print(f"\n=== {model_type.upper()} MODEL EVALUATION ===")
    for i, (input_text, target_text, prediction) in enumerate(zip(input_texts, target_texts, predictions)):
        reference = target_text.lower().split()
        prediction_tokens = prediction.lower().split() if prediction else ['']
        
        bleu_score = sentence_bleu([reference], prediction_tokens, smoothing_function=smoothing_fn)
        bleu_scores.append(bleu_score)
        
        print(f"\nExample {i+1}:")
        print(f"Input (French): {input_text}")
        print(f"Target (English): {target_text}")
        print(f"Prediction: {prediction}")
        print(f"BLEU Score: {bleu_score:.4f}")
    
    avg_bleu = np.mean(bleu_scores)
    print(f"\nAverage BLEU Score: {avg_bleu:.4f}")
    
    return avg_bleu

if __name__ == "__main__":
    # Evaluate both models
    try:
        basic_bleu = evaluate_model("basic")
    except Exception as e:
        print(f"Error evaluating basic model: {e}")
        basic_bleu = 0
    
    try:
        attention_bleu = evaluate_model("attention")
    except Exception as e:
        print(f"Error evaluating attention model: {e}")
        attention_bleu = 0
    
    print(f"\n=== FINAL COMPARISON ===")
    print(f"Basic Seq2Seq BLEU: {basic_bleu:.4f}")
    print(f"Attention Seq2Seq BLEU: {attention_bleu:.4f}")



    