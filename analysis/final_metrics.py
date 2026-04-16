"""Final metrics calculation for cascade failure study."""

import json
import os
from typing import Dict, List, Tuple
from scipy import stats
import numpy as np


def load_results() -> Dict:
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


def calculate_chain_a_metrics(results: List[dict]) -> Dict:
    """Calculate metrics for Chain A (2-step weather chain)."""
    total = len(results)
    
    # Syntactic success (valid JSON)
    syntactic = sum(1 for r in results if r.get('step1_syntactic_success', False))
    
    # Semantic success (correct location)
    semantic = sum(1 for r in results if r.get('step1_semantic_success', False))
    
    # Silent failures (valid JSON but wrong location)
    silent = syntactic - semantic
    
    # Semantic cascade (wrong location propagates to step 2)
    sem_cascade = sum(1 for r in results if r.get('semantic_cascade', False))
    
    # Traditional cascade failure
    cascade = sum(1 for r in results if r.get('cascade_failure', False))
    
    # Complete success (both steps work AND semantic correct)
    complete = sum(1 for r in results if r.get('step1_success', False) and r.get('step2_success', False) and r.get('step1_semantic_success', False))
    
    return {
        'total': total,
        'syntactic_success': syntactic,
        'semantic_success': semantic,
        'silent_failures': silent,
        'semantic_cascades': sem_cascade,
        'cascade_failures': cascade,
        'complete_success': complete,
        'syntactic_ssr': syntactic / total * 100,
        'semantic_ssr': semantic / total * 100,
        'silent_failure_rate': silent / total * 100,
        'semantic_cascade_rate': sem_cascade / total * 100,
        'cascade_failure_rate': cascade / total * 100,
        'complete_success_rate': complete / total * 100,
    }


def classify_failure_mode(result: dict) -> str:
    """Classify a result into a failure mode."""
    syntactic = result.get('step1_syntactic_success', False)
    semantic = result.get('step1_semantic_success', False)
    step2 = result.get('step2_success', False)
    
    # Success: Both steps work, semantic correct
    if syntactic and semantic and step2:
        return 'success'
    
    # Silent corruption: Valid JSON, wrong parameters, step 2 still works
    if syntactic and not semantic:
        return 'silent_corruption'
    
    # Hard failure: Invalid JSON or step failure
    return 'hard_failure'


def calculate_failure_mode_distribution(results: List[dict]) -> Dict:
    """Calculate distribution of failure modes."""
    modes = {'success': 0, 'silent_corruption': 0, 'hard_failure': 0}
    
    for r in results:
        mode = classify_failure_mode(r)
        modes[mode] += 1
    
    total = len(results)
    return {
        mode: count / total * 100 
        for mode, count in modes.items()
    }


def calculate_chain_bc_metrics(results: List[dict]) -> Dict:
    """Calculate metrics for Chain B/C (multi-step)."""
    total = len(results)
    
    # End-to-end success (all steps successful)
    if 'all_steps_successful' in results[0]:
        e2e_success = sum(1 for r in results if r.get('all_steps_successful', False))
    else:
        e2e_success = sum(1 for r in results if all(s.get('success', False) for s in r.get('steps', [])))
    
    # Cascade failure (any step fails)
    cascade = sum(1 for r in results if r.get('cascade_failure', False))
    
    # Step-by-step success rates
    num_steps = len(results[0].get('steps', [])) if results else 0
    step_success = {}
    for i in range(num_steps):
        successes = sum(1 for r in results if len(r.get('steps', [])) > i and r['steps'][i].get('success', False))
        step_success[f'step_{i+1}'] = {
            'success': successes,
            'rate': successes / total * 100
        }
    
    return {
        'total': total,
        'e2e_success': e2e_success,
        'cascade_failure': cascade,
        'e2e_success_rate': e2e_success / total * 100,
        'cascade_failure_rate': cascade / total * 100,
        'step_success': step_success,
    }


def welch_anova_test(groups: List[List[float]]) -> Dict:
    """Perform Welch's ANOVA test."""
    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return {'error': 'Need at least 2 groups'}
    
    f_stat, p_value = stats.f_oneway(*groups)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': 'Significant difference between groups' if p_value < 0.05 else 'No significant difference'
    }


def calculate_effect_size(groups: List[List[float]]) -> Dict:
    """Calculate effect size (eta-squared)."""
    # Combine all groups
    all_values = []
    group_means = []
    
    for g in groups:
        if len(g) > 0:
            all_values.extend(g)
            group_means.append(np.mean(g))
    
    if len(all_values) < 2:
        return {'error': 'Insufficient data'}
    
    grand_mean = np.mean(all_values)
    
    # Between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups if len(g) > 0)
    
    # Total sum of squares
    ss_total = sum((x - grand_mean)**2 for x in all_values)
    
    # Eta-squared
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    # Effect size interpretation
    if eta_squared < 0.01:
        interpretation = 'negligible'
    elif eta_squared < 0.06:
        interpretation = 'small'
    elif eta_squared < 0.14:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return {
        'eta_squared': eta_squared,
        'interpretation': interpretation,
        'ss_between': ss_between,
        'ss_total': ss_total,
    }


def generate_report():
    """Generate final metrics report."""
    results = load_results()
    
    report = []
    report.append("=" * 80)
    report.append("FINAL METRICS REPORT - CASCADE FAILURE IN LLM TOOL CHAINS")
    report.append("=" * 80)
    
    # ============ Chain A: Three Model Comparison ============
    report.append("\n" + "=" * 80)
    report.append("CHAIN A (2-STEP): THREE MODEL COMPARISON")
    report.append("=" * 80)
    
    chain_a_metrics = {}
    for model in ['qwen', 'llama', 'phi3']:
        if model in results:
            metrics = calculate_chain_a_metrics(results[model])
            chain_a_metrics[model] = metrics
            
            report.append(f"\n--- {model.upper()} ---")
            report.append(f"Total Instances: {metrics['total']}")
            report.append(f"Syntactic SSR: {metrics['syntactic_ssr']:.1f}% ({metrics['syntactic_success']}/{metrics['total']})")
            report.append(f"Semantic SSR: {metrics['semantic_ssr']:.1f}% ({metrics['semantic_success']}/{metrics['total']})")
            report.append(f"Silent Failures: {metrics['silent_failure_rate']:.1f}% ({metrics['silent_failures']}/{metrics['total']})")
            report.append(f"Semantic Cascade Rate: {metrics['semantic_cascade_rate']:.1f}% ({metrics['semantic_cascades']}/{metrics['total']})")
            report.append(f"Traditional CFR: {metrics['cascade_failure_rate']:.1f}% ({metrics['cascade_failures']}/{metrics['total']})")
            report.append(f"Complete Success: {metrics['complete_success_rate']:.1f}% ({metrics['complete_success']}/{metrics['total']})")
    
    # Failure mode distribution
    report.append("\n--- FAILURE MODE DISTRIBUTION ---")
    for model in ['qwen', 'llama', 'phi3']:
        if model in results:
            modes = calculate_failure_mode_distribution(results[model])
            report.append(f"{model.upper()}:")
            report.append(f"  Success: {modes['success']:.1f}%")
            report.append(f"  Silent Corruption: {modes['silent_corruption']:.1f}%")
            report.append(f"  Hard Failure: {modes['hard_failure']:.1f}%")
    
    # ============ Chain B/C: Multi-step Analysis ============
    report.append("\n" + "=" * 80)
    report.append("CHAIN B/C (MULTI-STEP): CASCADE BY CHAIN LENGTH")
    report.append("=" * 80)
    
    for chain in ['chain_b', 'chain_c']:
        if chain in results:
            metrics = calculate_chain_bc_metrics(results[chain])
            
            report.append(f"\n--- {chain.upper()} ---")
            report.append(f"Steps: {len(metrics['step_success'])}")
            report.append(f"E2E Success Rate: {metrics['e2e_success_rate']:.1f}%")
            report.append(f"Cascade Failure Rate: {metrics['cascade_failure_rate']:.1f}%")
            
            report.append("Step-by-Step Success:")
            for step, data in metrics['step_success'].items():
                report.append(f"  {step}: {data['rate']:.1f}% ({data['success']}/{metrics['total']})")
    
    # ============ Statistical Tests ============
    report.append("\n" + "=" * 80)
    report.append("STATISTICAL TESTS")
    report.append("=" * 80)
    
    # Semantic cascade rates for ANOVA
    scr_groups = [
        [1 if r.get('semantic_cascade', False) else 0 for r in results.get('qwen', [])],
        [1 if r.get('semantic_cascade', False) else 0 for r in results.get('llama', [])],
        [1 if r.get('semantic_cascade', False) else 0 for r in results.get('phi3', [])],
    ]
    
    anova_result = welch_anova_test(scr_groups)
    report.append(f"\nWelch's ANOVA (Semantic Cascade across models):")
    report.append(f"  F-statistic: {anova_result.get('f_statistic', 'N/A')}")
    report.append(f"  p-value: {anova_result.get('p_value', 'N/A')}")
    report.append(f"  Significant: {anova_result.get('significant', 'N/A')}")
    
    effect = calculate_effect_size(scr_groups)
    report.append(f"\nEffect Size (Eta-squared):")
    report.append(f"  η² = {effect.get('eta_squared', 'N/A'):.4f}")
    report.append(f"  Interpretation: {effect.get('interpretation', 'N/A')}")
    
    # ============ Summary Table ============
    report.append("\n" + "=" * 80)
    report.append("SUMMARY TABLE")
    report.append("=" * 80)
    
    report.append(f"\n{'Model':<15} {'Chain':<10} {'SynSSR':<10} {'SemSSR':<10} {'SCR':<10} {'CFR':<10}")
    report.append("-" * 65)
    
    # Chain A
    for model, metrics in chain_a_metrics.items():
        report.append(f"{model.upper():<15} {'A (2-step)':<10} {metrics['syntactic_ssr']:<10.1f} {metrics['semantic_ssr']:<10.1f} {metrics['semantic_cascade_rate']:<10.1f} {metrics['cascade_failure_rate']:<10.1f}")
    
    # Chain B
    if 'chain_b' in results:
        b_metrics = calculate_chain_bc_metrics(results['chain_b'])
        report.append(f"{'Qwen'.upper():<15} {'B (3-step)':<10} {b_metrics['step_success']['step_1']['rate']:<10.1f} {'N/A':<10} {'N/A':<10} {b_metrics['cascade_failure_rate']:<10.1f}")
    
    # Chain C
    if 'chain_c' in results:
        c_metrics = calculate_chain_bc_metrics(results['chain_c'])
        report.append(f"{'Qwen'.upper():<15} {'C (5-step)':<10} {c_metrics['step_success']['step_1']['rate']:<10.1f} {'N/A':<10} {'N/A':<10} {c_metrics['cascade_failure_rate']:<10.1f}")
    
    # Print report
    report_text = '\n'.join(report)
    print(report_text)
    
    # Save to file
    os.makedirs('results', exist_ok=True)
    with open('results/final_metrics_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Report saved to results/final_metrics_report.txt")
    
    # Return structured data for figure generation
    return {
        'chain_a_metrics': chain_a_metrics,
        'chain_b_metrics': calculate_chain_bc_metrics(results.get('chain_b', [])),
        'chain_c_metrics': calculate_chain_bc_metrics(results.get('chain_c', [])),
        'statistical_tests': {
            'welch_anova': anova_result,
            'effect_size': effect,
        }
    }


if __name__ == "__main__":
    generate_report()