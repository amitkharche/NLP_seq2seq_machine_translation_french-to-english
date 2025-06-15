
---
# 🌍 French-to-English Neural Machine Translation 🇫🇷→🇬🇧

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧠 Project Overview

This project demonstrates **Neural Machine Translation (NMT)** from **French to English** using:

- 📦 Basic **Seq2Seq model** (Encoder-Decoder LSTM)
- 💡 **Attention-enhanced Seq2Seq model**
- 🎯 **BLEU Score Evaluation**
- 🌐 **Streamlit Web App** for interactive translation

---

## 📁 Folder Structure

```

NLP\_seq2seq\_machine\_translation\_french-to-english/
│
├── scripts/                         # Training & evaluation scripts
│   ├── train\_basic\_seq2seq.py
│   ├── train\_attention\_seq2seq.py
│   └── evaluate\_bleu.py
│
├── streamlit\_app/                   # Streamlit interface
│   └── app.py
│
├── utils/                           # Utilities
│   ├── decode\_utils.py
│   ├── tokenizer\_utils.py
│   └── attention\_plot.py
│
├── data/                            # Dataset
│   └── french\_english\_pairs.txt
│
├── models/                          # Saved models & tokenizers
│   ├── basic\_encoder.h5
│   ├── basic\_decoder.h5
│   ├── attention\_seq2seq.h5
│   ├── tokenizer\_in.pkl
│   └── tokenizer\_out.pkl
│
├── README.md
└── requirements.txt

```

---

## 📄 Dataset

- **Location**: `data/french_english_pairs.txt`
- **Format**: Tab-separated French–English sentence pairs
```

bonjour	hello
merci beaucoup	thank you very much

````

---

## ⚙️ Setup Instructions

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/amitkharche/NLP_seq2seq_machine_translation_french-to-english.git
cd NLP_seq2seq_machine_translation_french-to-english
````

### ✅ 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
```

### ✅ 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If the file is missing, install manually:

```bash
pip install tensorflow streamlit nltk matplotlib seaborn
```

---

## 🚀 How to Use

### 🔹 Step 1: Train the Models

#### ➤ Basic Seq2Seq

```bash
python scripts/train_basic_seq2seq.py
```

#### ➤ Attention Seq2Seq

```bash
python scripts/train_attention_seq2seq.py
```

This saves the models under `models/`:

* `basic_seq2seq.h5`, `basic_encoder.h5`, `basic_decoder.h5`
* `attention_seq2seq.h5`
* `tokenizer_in.pkl`, `tokenizer_out.pkl`

---

### 🔹 Step 2: Evaluate BLEU Score

```bash
python scripts/evaluate_bleu.py
```

You'll get scores like:

```
=== BASIC MODEL EVALUATION ===
BLEU Score: 0.6701

=== ATTENTION MODEL EVALUATION ===
BLEU Score: 0.8154

=== FINAL COMPARISON ===
Basic Seq2Seq BLEU: 0.6701
Attention Seq2Seq BLEU: 0.8154
```

---

### 🔹 Step 3: Launch Streamlit App

```bash
streamlit run streamlit_app/app.py
```

Features:

* Choose between **Basic** or **Attention** model
* Input any French sentence or use preloaded examples
* See translation + optional attention heatmap
* Save your translations
* Debug and view model/tokenizer status

---

## 🎯 Key Features

| Feature             | Description                                        |
| ------------------- | -------------------------------------------------- |
| ✅ Basic Seq2Seq     | Standard LSTM Encoder-Decoder model                |
| ✨ Attention Layer   | Custom layer for dynamic word-level context        |
| 📊 BLEU Evaluation  | Objective translation quality comparison           |
| 🌐 Streamlit UI     | Easy interface to test translation results         |
| 📈 Visual Attention | Simulated attention heatmap for demo understanding |

---

## 📌 Requirements

* Python ≥ 3.8
* TensorFlow ≥ 2.8
* Streamlit ≥ 1.22
* NLTK, NumPy, Matplotlib, Seaborn

---

## 🛠️ Improvements You Can Make

* Add **beam search** decoding for better outputs
* Fine-tune attention weights to support **live visualization**
* Use **transformer architectures** (e.g. BERT2BERT, MarianMT)
* Integrate **multi-language translation** features

---

## 📬 Let's Connect

Have questions or want to collaborate?

* 🔗 [LinkedIn](https://www.linkedin.com/in/amit-kharche)
* 🧠 [Medium](https://medium.com/@amitkharche14)
* 💻 [GitHub](https://github.com/amitkharche)

---

## 📜 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---
