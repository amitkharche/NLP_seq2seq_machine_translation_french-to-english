import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_attention_weights(attention_weights, input_tokens, output_tokens):
    """Plot attention weights as heatmap"""
    plt.figure(figsize=(max(len(input_tokens), 8), max(len(output_tokens), 6)))
    
    # Ensure dimensions match
    min_rows = min(len(output_tokens), attention_weights.shape[0])
    min_cols = min(len(input_tokens), attention_weights.shape[1])
    
    sns.heatmap(attention_weights[:min_rows, :min_cols], 
                xticklabels=input_tokens[:min_cols],
                yticklabels=output_tokens[:min_rows],
                cmap='Blues',
                cbar=True,
                square=True)
    
    plt.xlabel('Input (French)')
    plt.ylabel('Output (English)')
    plt.title('Attention Weights Heatmap')
    plt.tight_layout()
    return plt.gcf()
