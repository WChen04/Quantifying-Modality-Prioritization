"""
Generate figures for the research paper.
Creates publication-ready plots for LaTeX inclusion.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

def load_results(mode):
    """Load results CSV."""
    path = f"results/experiment_{mode}.csv"
    return pd.read_csv(path)


def figure1_asr_comparison():
    """Figure 1: ASR Comparison across modes."""
    modes = ['text', 'visual', 'omni']
    results = {}
    
    for mode in modes:
        try:
            df = load_results(mode)
            valid = df[df['score'] >= 0]
            asr = (valid['score'] == 1).mean() * 100
            results[mode] = asr
        except:
            results[mode] = 0
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    bars = ax.bar(
        ['Text-Only', 'Visual-Only', 'Omni-Modal'],
        [results['text'], results['visual'], results['omni']],
        color=['#3498db', '#e74c3c', '#2ecc71'],
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Attack Success Rate (%)', fontweight='bold')
    ax.set_xlabel('Attack Vector', fontweight='bold')
    ax.set_title('Attack Success Rates Across Modalities', fontweight='bold', pad=20)
    ax.set_ylim(0, max(results.values()) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('figures/fig1_asr_comparison.pdf', bbox_inches='tight')
    plt.savefig('figures/fig1_asr_comparison.png', bbox_inches='tight', dpi=300)
    print("✓ Figure 1 saved: ASR Comparison")


def figure2_category_breakdown():
    """Figure 2: ASR by attack category."""
    try:
        df_omni = load_results('omni')
        df_visual = load_results('visual')
        
        categories = df_omni['category'].unique()
        omni_asrs = []
        visual_asrs = []
        
        for cat in categories:
            omni_cat = df_omni[(df_omni['category'] == cat) & (df_omni['score'] >= 0)]
            visual_cat = df_visual[(df_visual['category'] == cat) & (df_visual['score'] >= 0)]
            
            omni_asrs.append((omni_cat['score'] == 1).mean() * 100 if len(omni_cat) > 0 else 0)
            visual_asrs.append((visual_cat['score'] == 1).mean() * 100 if len(visual_cat) > 0 else 0)
        
        # Sort by omni ASR
        sorted_data = sorted(zip(categories, omni_asrs, visual_asrs), key=lambda x: x[1], reverse=True)
        categories, omni_asrs, visual_asrs = zip(*sorted_data)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, visual_asrs, width, label='Visual-Only',
                       color='#e74c3c', edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, omni_asrs, width, label='Omni-Modal',
                       color='#2ecc71', edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Attack Success Rate (%)', fontweight='bold')
        ax.set_xlabel('Attack Category', fontweight='bold')
        ax.set_title('Audio Multiplier Effect by Attack Category', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend(frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('figures/fig2_category_breakdown.pdf', bbox_inches='tight')
        plt.savefig('figures/fig2_category_breakdown.png', bbox_inches='tight', dpi=300)
        print("✓ Figure 2 saved: Category Breakdown")
        
    except Exception as e:
        print(f"⚠ Could not generate Figure 2: {e}")


def figure3_multiplier_effect():
    """Figure 3: Audio Multiplier visualization."""
    try:
        df_visual = load_results('visual')
        df_omni = load_results('omni')
        
        valid_v = df_visual[df_visual['score'] >= 0]
        valid_o = df_omni[df_omni['score'] >= 0]
        
        visual_asr = (valid_v['score'] == 1).mean() * 100
        omni_asr = (valid_o['score'] == 1).mean() * 100
        multiplier = (omni_asr / visual_asr) if visual_asr > 0 else 0
        gain = omni_asr - visual_asr
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Left: Multiplier
        ax1.bar(['Visual\nBaseline', 'Audio\nMultiplier'], 
                [1.0, multiplier],
                color=['#e74c3c', '#2ecc71'],
                edgecolor='black',
                linewidth=1.5)
        ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_ylabel('Relative ASR Multiplier', fontweight='bold')
        ax1.set_title('Audio Multiplier Effect', fontweight='bold')
        ax1.text(1, multiplier, f'{multiplier:.2f}x', ha='center', va='bottom', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Right: Absolute gain
        ax2.bar(['Visual-Only', 'Omni-Modal'],
                [visual_asr, omni_asr],
                color=['#e74c3c', '#2ecc71'],
                edgecolor='black',
                linewidth=1.5)
        
        # Add arrow showing gain
        ax2.annotate('', xy=(1, omni_asr), xytext=(1, visual_asr),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax2.text(1.1, (visual_asr + omni_asr)/2, f'+{gain:.1f}pp',
                fontweight='bold', va='center')
        
        ax2.set_ylabel('Attack Success Rate (%)', fontweight='bold')
        ax2.set_title('Absolute ASR Increase', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('figures/fig3_multiplier_effect.pdf', bbox_inches='tight')
        plt.savefig('figures/fig3_multiplier_effect.png', bbox_inches='tight', dpi=300)
        print("✓ Figure 3 saved: Multiplier Effect")
        
    except Exception as e:
        print(f"⚠ Could not generate Figure 3: {e}")


def main():
    """Generate all figures."""
    # Create figures directory
    Path("figures").mkdir(exist_ok=True)
    
    print("Generating publication-ready figures...")
    print("-" * 50)
    
    figure1_asr_comparison()
    figure2_category_breakdown()
    figure3_multiplier_effect()
    
    print("-" * 50)
    print("✅ All figures generated in figures/ directory")
    print("\nInclude in LaTeX with:")
    print("  \\includegraphics[width=0.8\\textwidth]{figures/fig1_asr_comparison.pdf}")


if __name__ == "__main__":
    main()

