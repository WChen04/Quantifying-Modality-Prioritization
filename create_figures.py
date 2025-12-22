"""
Generate figures for the research paper.
Creates publication-ready plots with heatmaps and dual-architecture comparisons.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.5

def load_results(directory, mode):
    """Load results CSV from specified directory."""
    path = f"{directory}/experiment_{mode}.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Warning: {path} not found")
        return None


def calculate_asr(df):
    """Calculate Attack Success Rate from DataFrame."""
    if df is None or len(df) == 0:
        return 0.0
    valid = df[df['score'] >= 0]
    if len(valid) == 0:
        return 0.0
    return (valid['score'] == 1).mean() * 100


def figure1_dual_architecture_comparison():
    """Figure 1: Dual-architecture ASR comparison with quantification metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Load data for both architectures
    llava_results = {}
    qwen_results = {}
    
    for mode in ['text', 'visual', 'omni']:
        llava_results[mode] = calculate_asr(load_results('results_llava', mode))
        qwen_results[mode] = calculate_asr(load_results('results', mode))
    
    # Calculate prioritization indices
    llava_delta = ((llava_results['omni'] - llava_results['visual']) / llava_results['visual'] * 100) if llava_results['visual'] > 0 else 0
    qwen_delta = ((qwen_results['omni'] - qwen_results['visual']) / qwen_results['visual'] * 100) if qwen_results['visual'] > 0 else 0
    
    # Prepare data
    modes = ['Text-Only', 'Visual-Only', 'Omni-Modal']
    x = np.arange(len(modes))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, 
                   [llava_results['text'], llava_results['visual'], llava_results['omni']], 
                   width, 
                   label='LLaVA-v1.6 + Whisper',
                   color='#3498db', 
                   edgecolor='black', 
                   linewidth=1.5,
                   alpha=0.85)
    
    bars2 = ax.bar(x + width/2, 
                   [qwen_results['text'], qwen_results['visual'], qwen_results['omni']], 
                   width, 
                   label='Qwen2-Audio-7B',
                   color='#e74c3c', 
                   edgecolor='black', 
                   linewidth=1.5,
                   alpha=0.85)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add prioritization index annotations
    ax.text(0.5, max(llava_results['text'], qwen_results['text']) + 3,
            f'LLaVA Δ_audio = {llava_delta:.1f}%\nQwen2 Δ_audio = {qwen_delta:.1f}%',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Attack Vector', fontweight='bold', fontsize=12)
    ax.set_title('Quantifying Modality Prioritization Across Architectures', fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend(frameon=True, shadow=True, fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max([llava_results['text'], llava_results['visual'], llava_results['omni'],
                         qwen_results['text'], qwen_results['visual'], qwen_results['omni']]) * 1.25)
    
    plt.tight_layout()
    plt.savefig('figures/fig1_asr_comparison.pdf', bbox_inches='tight')
    plt.savefig('figures/fig1_asr_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Figure 1 saved: Dual-Architecture ASR Comparison")


def figure2_category_heatmap():
    """Figure 2: Professional heatmap showing category-specific vulnerabilities."""
    # Load data
    llava_text = load_results('results_llava', 'text')
    llava_visual = load_results('results_llava', 'visual')
    llava_omni = load_results('results_llava', 'omni')
    
    qwen_text = load_results('results', 'text')
    qwen_visual = load_results('results', 'visual')
    qwen_omni = load_results('results', 'omni')
    
    # Get all categories
    if llava_visual is not None:
        categories = sorted(llava_visual['category'].unique())
    else:
        categories = ['Weapons', 'Fraud', 'Malware', 'Drugs', 'Violence', 'Harassment', 'Privacy', 'Illegal', 'CSAM']
    
    # Build data matrix: [Categories x Conditions]
    # Rows: Categories, Columns: LLaVA Text, LLaVA Visual, LLaVA Omni, Qwen2 Text, Qwen2 Visual, Qwen2 Omni
    data = []
    
    for cat in categories:
        row = []
        
        # LLaVA results
        for df in [llava_text, llava_visual, llava_omni]:
            if df is not None:
                cat_df = df[(df['category'] == cat) & (df['score'] >= 0)]
                asr = (cat_df['score'] == 1).mean() * 100 if len(cat_df) > 0 else 0
                row.append(asr)
            else:
                row.append(0)
        
        # Qwen2 results
        for df in [qwen_text, qwen_visual, qwen_omni]:
            if df is not None:
                cat_df = df[(df['category'] == cat) & (df['score'] >= 0)]
                asr = (cat_df['score'] == 1).mean() * 100 if len(cat_df) > 0 else 0
                row.append(asr)
            else:
                row.append(0)
        
        data.append(row)
    
    # Create DataFrame for heatmap
    df_heatmap = pd.DataFrame(
        data,
        index=categories,
        columns=['LLaVA\nText', 'LLaVA\nVisual', 'LLaVA\nOmni', 
                 'Qwen2\nText', 'Qwen2\nVisual', 'Qwen2\nOmni']
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use diverging colormap: white (0%) → dark red (high ASR)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    sns.heatmap(df_heatmap, 
                annot=True, 
                fmt='.1f', 
                cmap=cmap,
                linewidths=2, 
                linecolor='black',
                cbar_kws={'label': 'Attack Success Rate (%)', 'shrink': 0.8},
                vmin=0,
                vmax=max(50, df_heatmap.max().max()),  # Scale to at least 50% for visibility
                ax=ax,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    ax.set_title('Category-Specific Vulnerability Heatmap', fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('Architecture & Condition', fontweight='bold', fontsize=12)
    ax.set_ylabel('Threat Category', fontweight='bold', fontsize=12)
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('figures/fig2_category_breakdown.pdf', bbox_inches='tight')
    plt.savefig('figures/fig2_category_breakdown.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Figure 2 saved: Category-Specific Vulnerability Heatmap")


def figure3_prioritization_index():
    """Figure 3: Prioritization Index visualization."""
    # Load data
    llava_visual = load_results('results_llava', 'visual')
    llava_omni = load_results('results_llava', 'omni')
    qwen_visual = load_results('results', 'visual')
    qwen_omni = load_results('results', 'omni')
    
    # Calculate overall metrics
    llava_v_asr = calculate_asr(llava_visual)
    llava_o_asr = calculate_asr(llava_omni)
    qwen_v_asr = calculate_asr(qwen_visual)
    qwen_o_asr = calculate_asr(qwen_omni)
    
    # Calculate prioritization indices
    llava_delta = ((llava_o_asr - llava_v_asr) / llava_v_asr * 100) if llava_v_asr > 0 else 0
    qwen_delta = ((qwen_o_asr - qwen_v_asr) / qwen_v_asr * 100) if qwen_v_asr > 0 else 0
    
    # Calculate absolute shifts
    llava_shift = llava_o_asr - llava_v_asr
    qwen_shift = qwen_o_asr - qwen_v_asr
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Prioritization Index (Relative)
    architectures = ['LLaVA\n(Pipeline)', 'Qwen2\n(Native)']
    deltas = [llava_delta, qwen_delta]
    colors = ['#3498db' if d < 0 else '#e74c3c' if d > 0 else '#95a5a6' for d in deltas]
    
    bars1 = ax1.bar(architectures, deltas, color=colors, edgecolor='black', linewidth=2, alpha=0.85)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax1.set_ylabel('Prioritization Index Δ_audio (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Prioritization Index:\nRelative Audio Influence', fontweight='bold', fontsize=13, pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, delta) in enumerate(zip(bars1, deltas)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (2 if height > 0 else -2),
                f'{delta:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold', fontsize=11)
    
    # Add interpretation labels
    ax1.text(0, min(deltas) - 10, 'Audio\nPrioritized', ha='center', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Right plot: Absolute ASR Shift
    x = np.arange(len(architectures))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, [llava_v_asr, qwen_v_asr], width, 
                    label='Visual-Only', color='#e74c3c', edgecolor='black', linewidth=1.5)
    bars3 = ax2.bar(x + width/2, [llava_o_asr, qwen_o_asr], width, 
                    label='Omni-Modal', color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
    ax2.set_title('Absolute ASR Shift:\nVisual vs. Omni-Modal', fontweight='bold', fontsize=13, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(architectures)
    ax2.legend(frameon=True, shadow=True, fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add shift annotations
    for i, (v_asr, o_asr, shift) in enumerate([(llava_v_asr, llava_o_asr, llava_shift), 
                                                 (qwen_v_asr, qwen_o_asr, qwen_shift)]):
        if shift != 0:
            ax2.annotate('', xy=(i, o_asr), xytext=(i, v_asr),
                        arrowprops=dict(arrowstyle='<->', color='black', lw=2))
            ax2.text(i + 0.2, (v_asr + o_asr)/2, 
                    f'{shift:+.1f}pp',
                    fontweight='bold', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/fig3_multiplier_effect.pdf', bbox_inches='tight')
    plt.savefig('figures/fig3_multiplier_effect.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Figure 3 saved: Prioritization Index Visualization")


def main():
    """Generate all figures."""
    # Create figures directory
    Path("figures").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Generating Publication-Ready Figures with Heatmaps")
    print("=" * 60)
    
    figure1_dual_architecture_comparison()
    figure2_category_heatmap()
    figure3_prioritization_index()
    
    print("=" * 60)
    print("All figures generated in figures/ directory")
    print("=" * 60)
    print("\nFiles created:")
    print("  - fig1_asr_comparison.pdf/.png (Dual-architecture comparison)")
    print("  - fig2_category_breakdown.pdf/.png (Category heatmap)")
    print("  - fig3_multiplier_effect.pdf/.png (Prioritization index)")
    print("\nInclude in LaTeX with:")
    print("  \\includegraphics[width=0.9\\columnwidth]{figures/fig1_asr_comparison.pdf}")


if __name__ == "__main__":
    main()
