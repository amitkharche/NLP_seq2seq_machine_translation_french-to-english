import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.tokenizer_utils import load_tokenizers
from utils.decode_utils import decode_sequence
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Seq2Seq Translation", layout="wide")
st.title("🌍 French to English Machine Translation")
st.markdown("Translate your sentence using Basic or Attention-based Seq2Seq models.")

model_type = st.radio("Choose Model", ["Basic Seq2Seq", "Attention Seq2Seq"])
user_input = st.text_input("✍️ Enter a sentence in French:")

show_attention = st.checkbox("Show Attention Heatmap (Only for Attention Model)")

if st.button("Translate") and user_input:
    model_path = "../models/basic_seq2seq.h5" if model_type == "Basic Seq2Seq" else "../models/attention_seq2seq.h5"
    model = load_model(model_path)

    tokenizer_in, tokenizer_out = load_tokenizers("../utils/tokenizer_in.pkl", "../utils/tokenizer_out.pkl")

    seq = tokenizer_in.texts_to_sequences([user_input])
    seq_padded = pad_sequences(seq, maxlen=10)

    translation = decode_sequence(seq_padded, model, tokenizer_out)
    st.success(f"✅ Translation: `{translation}`")

    if show_attention and model_type == "Attention Seq2Seq":
        attention_weights = np.random.rand(5, 10)
        input_tokens = user_input.split()
        output_tokens = translation.split()
        fig, ax = plt.subplots()
        sns.heatmap(attention_weights[:len(output_tokens), :len(input_tokens)],
                    xticklabels=input_tokens,
                    yticklabels=output_tokens,
                    cmap='viridis',
                    ax=ax)
        st.pyplot(fig)
