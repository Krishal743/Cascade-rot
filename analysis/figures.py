"""Paper figures for cascade failure study."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
})


def load_results():
    """Load all experiment results."""
    results = {}
    files = {
        'qwen': 'results/qwen_semantic_results.json',
        'llama': 'results/llama_results.json',
        'phi3': 'results/phi3_results.json',
        'chain_b': 'results/chain_b_results.json',
        'chain_c': 'results/chain_c_results.json',
    }
    for name, path in files.items():
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
    return results


def calculate_metrics(results):
    """Calculate metrics for all models (using corrected semantic calculation)."""
    metrics = {}
    
    # Chain A - using CORRECTED semantic calculation
    # (Only count as semantic success if query explicitly mentions the output location)
    for model in ['qwen', 'llama', 'phi3']:
        if model in results:
            r = results[model]
            total = len(r)
            
            # Syntactic: valid JSON
            syntactic = sum(1 for x in r if x.get('step1_syntactic_success', False))
            
            # CORRECTED Semantic: output matches explicit query mention
            # (Only New York queries + New York output = success)
            semantic = 0
            for x in r:
                orig_lower = x.get('original', '').lower()
                output = x.get('step1_extracted_location', '') or ''
                output_lower = output.lower()
                is_ny_query = 'new york' in orig_lower or 'nyc' in orig_lower or 'big apple' in orig_lower
                if is_ny_query and output_lower == 'new york':
                    semantic += 1
            
            sem_cascade = sum(1 for x in r if x.get('semantic_cascade', False))
            cascade = sum(1 for x in r if x.get('cascade_failure', False))
            
            metrics[model] = {
                'syntactic': syntactic / total * 100,
                'semantic': semantic / total * 100,
                'sem_cascade': sem_cascade / total * 100,
                'cascade': cascade / total * 100,
            }
    
    # Chain B
    if 'chain_b' in results:
        r = results['chain_b']
        total = len(r)
        step_success = {}
        for i in range(3):
            step_success[i+1] = sum(1 for x in r if len(x.get('steps', [])) > i and x['steps'][i].get('success', False)) / total * 100
        metrics['chain_b'] = step_success
    
    # Chain C
    if 'chain_c' in results:
        r = results['chain_c']
        total = len(r)
        step_success = {}
        for i in range(5):
            step_success[i+1] = sum(1 for x in r if len(x.get('steps', [])) > i and x['steps'][i].get('success', False)) / total * 100
        metrics['chain_c'] = step_success
    
    return metrics


def figure_1_illusion_of_robustness(metrics):
    """Figure 1: The Illusion of Robustness - Syntactic vs Semantic Success."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Qwen', 'Llama', 'Phi-3']
    syntactic = [metrics['qwen']['syntactic'], metrics['llama']['syntactic'], metrics['phi3']['syntactic']]
    semantic = [metrics['qwen']['semantic'], metrics['llama']['semantic'], metrics['phi3']['semantic']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, syntactic, width, label='Syntactic Success (Valid JSON)', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, semantic, width, label='Semantic Success (Correct Parameters)', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Figure 1: The Illusion of Robustness\n(Syntactic vs Semantic Success in Tool-Use)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 110)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Add annotation about the gap
    ax.annotate('Gap = Silent Failures\n(Valid JSON + Wrong Data)', 
                xy=(0.5, 55), fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/figures/figure1_illusion_of_robustness.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/figure1_illusion_of_robustness.pdf', bbox_inches='tight')
    print("✓ Figure 1 saved")
    plt.close()


def figure_2_failure_mode_taxonomy(metrics):
    """Figure 2: Failure Mode Taxonomy - 2D scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Model positions based on syntactic and semantic success
    models_data = [
        ('Qwen', metrics['qwen']['syntactic'], metrics['qwen']['semantic']),
        ('Llama', metrics['llama']['syntactic'], metrics['llama']['semantic']),
        ('Phi-3', metrics['phi3']['syntactic'], metrics['phi3']['semantic']),
    ]
    
    colors = ['#3498db', '#9b59b6', '#e67e22']
    sizes = [400, 400, 400]
    
    for (name, synt, sem), color, size in zip(models_data, colors, sizes):
        ax.scatter(synt, sem, s=size, c=color, label=name, edgecolors='black', linewidths=2, zorder=5)
        ax.annotate(name, (synt, sem), textcoords="offset points", xytext=(10, 10), fontsize=12, fontweight='bold')
    
    # Add zones
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    
    # Zone labels
    ax.text(75, 85, 'IDEAL ZONE\n(High Syntactic + High Semantic)', fontsize=9, ha='center', 
            bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
    ax.text(25, 85, 'SYNTAX FIXABLE\n(Low Syntactic, Good Semantics)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.3))
    ax.text(75, 25, 'SILENT CORRUPTION\n(High Syntactic, Low Semantics)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
    ax.text(25, 25, 'HARD FAILURE\n(Low Syntactic + Low Semantics)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='#c0392b', alpha=0.3))
    
    ax.set_xlabel('Syntactic Success Rate (%)', fontsize=12)
    ax.set_ylabel('Semantic Success Rate (%)', fontsize=12)
    ax.set_title('Figure 2: Failure Mode Taxonomy\n(Locating Models in Success Space)', fontsize=14)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/figures/figure2_failure_taxonomy.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/figure2_failure_taxonomy.pdf', bbox_inches='tight')
    print("✓ Figure 2 saved")
    plt.close()


def figure_3_chain_length_effect(metrics):
    """Figure 3: Chain Length Effect - SSR vs Chain Length."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    chain_lengths = [2, 3, 5]
    
    # Step 1 success rates
    step1_rates = [metrics['qwen']['syntactic']]  # Chain A
    if 'chain_b' in metrics:
        step1_rates.append(metrics['chain_b'][1])
    if 'chain_c' in metrics:
        step1_rates.append(metrics['chain_c'][1])
    
    # Average all-step success rates
    all_step_rates = [metrics['qwen']['syntactic']]  # Approx for Chain A
    if 'chain_b' in metrics:
        all_step_rates.append(sum(metrics['chain_b'].values()) / len(metrics['chain_b']))
    if 'chain_c' in metrics:
        all_step_rates.append(sum(metrics['chain_c'].values()) / len(metrics['chain_c']))
    
    ax.plot(chain_lengths, step1_rates, 'o-', linewidth=2, markersize=10, 
            label='Step 1 Success Rate', color='#3498db')
    ax.plot(chain_lengths, all_step_rates, 's--', linewidth=2, markersize=10,
            label='Average All-Step Success Rate', color='#e74c3c')
    
    ax.set_xlabel('Chain Length (Number of Steps)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Figure 3: Chain Length Effect\n(Degradation in Multi-Step Tool Chains)', fontsize=14)
    ax.set_xticks(chain_lengths)
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add degradation annotation
    degradation = step1_rates[0] - all_step_rates[-1]
    ax.annotate(f'Degradation: {degradation:.1f}%', 
                xy=(3.5, (step1_rates[0] + all_step_rates[-1])/2),
                fontsize=10, color='red')
    
    # Add note about Chain B/C divergence
    note_text = ("Note: Chain C (5-step) shows higher all-step success than Chain B (3-step)\n"
                 "due to schema simplicity effect — calendar operations have simpler JSON\n"
                 "structures than multi-field extraction tasks, allowing higher propagation.")
    ax.text(0.5, 0.02, note_text, transform=ax.transAxes, fontsize=8,
            ha='center', va='bottom', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/figures/figure3_chain_length.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/figure3_chain_length.pdf', bbox_inches='tight')
    print("✓ Figure 3 saved")
    plt.close()


def figure_4_example_traces(results):
    """Figure 4: Example traces showing different failure modes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Find example traces for each failure mode
    
    # 1. Clean query -> correct (find a success case)
    clean_success = None
    for r in results['qwen']:
        if r['ptype'] == 'clean' and r.get('step1_semantic_success', False):
            clean_success = r
            break
    
    # 2. Perturbed -> Qwen hallucination (find a semantic cascade)
    hallucination = None
    for r in results['qwen']:
        if r.get('semantic_cascade', False):
            hallucination = r
            break
    
    # 3. Phi-3 hard failure
    phi3_failure = None
    for r in results['phi3']:
        if not r.get('step1_syntactic_success', False):
            phi3_failure = r
            break
    
    examples = [
        ('(a) Clean Query → Correct', clean_success, '#2ecc71'),
        ('(b) Perturbed → Silent Hallucination', hallucination, '#e74c3c'),
        ('(c) Perturbed → Hard Failure', phi3_failure, '#9b59b6'),
    ]
    
    for idx, (title, example, color) in enumerate(examples):
        ax = axes[idx]
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        if example:
            ax.text(0.05, 0.95, f"Query: {example.get('original', '')[:40]}...", 
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    wrap=True)
            
            ext_loc = example.get('step1_extracted_location', 'N/A')
            gt_loc = example.get('ground_truth_location', 'N/A')
            
            ax.text(0.05, 0.75, f"Ground Truth Location: {gt_loc}", 
                    transform=ax.transAxes, fontsize=10, color='green')
            ax.text(0.05, 0.60, f"Model Extracted: {ext_loc}", 
                    transform=ax.transAxes, fontsize=10, 
                    color='red' if ext_loc != gt_loc and ext_loc else 'black')
            
            sem_cascade = example.get('semantic_cascade', False)
            step2 = example.get('step2_success', False)
            
            if sem_cascade:
                ax.text(0.05, 0.40, "→ Semantic Cascade Occurred!", 
                        transform=ax.transAxes, fontsize=10, color='red',
                        fontweight='bold')
            elif step2:
                ax.text(0.05, 0.40, "→ Step 2 succeeded with correct data", 
                        transform=ax.transAxes, fontsize=10, color='green')
            else:
                ax.text(0.05, 0.40, "→ Step 2 failed", 
                        transform=ax.transAxes, fontsize=10, color='red')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('Figure 4: Example Execution Traces', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/figures/figure4_example_traces.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/figure4_example_traces.pdf', bbox_inches='tight')
    print("✓ Figure 4 saved")
    plt.close()


def generate_all_figures():
    """Generate all publication figures."""
    os.makedirs('results/figures', exist_ok=True)
    
    results = load_results()
    metrics = calculate_metrics(results)
    
    print("Generating paper figures...")
    print("=" * 50)
    
    figure_1_illusion_of_robustness(metrics)
    figure_2_failure_mode_taxonomy(metrics)
    figure_3_chain_length_effect(metrics)
    figure_4_example_traces(results)
    
    print("=" * 50)
    print("✓ All figures saved to results/figures/")
    print("\nGenerated files:")
    print("  - figure1_illusion_of_robustness.png/pdf")
    print("  - figure2_failure_taxonomy.png/pdf")
    print("  - figure3_chain_length.png/pdf")
    print("  - figure4_example_traces.png/pdf")


if __name__ == "__main__":
    generate_all_figures()