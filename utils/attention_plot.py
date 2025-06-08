import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_attention_weights(attention_weights, input_tokens, output_tokens):
    plt.figure(figsize=(10, 6))
    sns.heatmap(attention_weights, xticklabels=input_tokens, yticklabels=output_tokens, cmap='viridis')
    plt.xlabel('Input (French)')
    plt.ylabel('Output (English)')
    plt.title('Attention Weights Heatmap')
    plt.show()
