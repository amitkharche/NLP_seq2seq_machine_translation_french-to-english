# 🌍 Seq2Seq Machine Translation – French to English ✨

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/github/license/your-username/seq2seq-machine-translation)

![Thumbnail](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Sequence_to_sequence_learning_diagram.svg/512px-Sequence_to_sequence_learning_diagram.svg.png)

---

## 📘 Overview
This project demonstrates how to build **neural machine translation** systems using **Sequence-to-Sequence (Seq2Seq)** architectures:

- ✅ Basic Encoder-Decoder Model (Keras Functional API)
- 🔁 Attention-enhanced Seq2Seq (Bahdanau attention)
- 🖥️ Streamlit Web App for live demo
- 📊 BLEU Score Evaluation
- 📚 Teaching Jupyter Notebooks
- 🧠 Pre-trained tokenizers and models
- 🚀 GitHub-ready codebase for learning and deployment

---

## 📂 Project Structure
```
seq2seq_machine_translation/
├── data/                     # Sample French-English dataset
├── models/                   # Saved models (basic + attention)
├── notebooks/                # Teaching notebooks (basic + attention)
├── scripts/                  # Training scripts
├── streamlit_app/            # Streamlit interface
├── utils/                    # Tokenizer + decoder helpers
├── output/                   # BLEU evaluation and attention plots
├── LICENSE
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Setup

```bash
git clone https://github.com/your-username/seq2seq-machine-translation.git
cd seq2seq-machine-translation
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🧠 Train Models

```bash
python scripts/train_with_tokenizer.py                # Train basic model
python scripts/train_attention_with_tokenizer.py      # Train attention model
```

### 🌐 Run Streamlit App

```bash
streamlit run streamlit_app/app.py
```

### 📏 BLEU Score Evaluation

```bash
python scripts/evaluate_bleu.py
```

---

## 🧪 Demo Screenshots

| Basic Seq2Seq | Attention Model |
|---------------|-----------------|
| ![Basic](https://i.imgur.com/J3hjRjA.png) | ![Attention](https://i.imgur.com/FZnU23I.png) |

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Credits
Built using:
- TensorFlow & Keras
- Streamlit
- NLTK BLEU
- Seaborn & Matplotlib

Inspired by:
- Stanford NLP
- OpenNMT

---

## 🧭 How to Run This Project (Step-by-Step)

### 🛠️ 1. Clone the Repository
```bash
git clone https://github.com/your-username/seq2seq-machine-translation.git
cd seq2seq-machine-translation
```

### 🧪 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 🧠 3. Train the Models

#### ▶️ Train the Basic Encoder-Decoder Model
```bash
python scripts/train_with_tokenizer.py
```

#### ▶️ Train the Attention-based Seq2Seq Model
```bash
python scripts/train_attention_with_tokenizer.py
```

> These scripts will also save the tokenizers used during preprocessing.

### 🧠 4. Run BLEU Score Evaluation
```bash
python scripts/evaluate_bleu.py
```

### 🌐 5. Launch the Streamlit Translation App
```bash
streamlit run streamlit_app/app.py
```

You can now enter a French sentence and:
- Select either the basic or attention-based model
- View translation output
- Optionally display the attention heatmap (if using attention-based model)

---

## ✅ Tips
- Trained models are saved to `models/` and loaded automatically by the app
- Tokenizers are stored as pickled files in `utils/`
- Sample data can be modified or extended in `data/french_english_pairs.txt`
