import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Dummy data (predictions and references)
references = [["this", "is", "a", "test"], ["hello", "world"]]
predictions = [["this", "is", "test"], ["hello", "there"]]

smoothing_fn = SmoothingFunction().method1
for ref, pred in zip(references, predictions):
    score = sentence_bleu([ref], pred, smoothing_function=smoothing_fn)
    print(f"Reference: {' '.join(ref)}")
    print(f"Prediction: {' '.join(pred)}")
    print(f"BLEU score: {score:.4f}\n")
