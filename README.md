# Brittle Semantic Collapse Under Perturbation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](link-to-your-paper)
[![Hardware](https://img.shields.io/badge/Hardware-RTX_3050_(6GB)-green.svg)]()

> **Official implementation and dataset for the paper:**
> *Brittle Semantic Collapse Under Perturbation: A Safety and Sustainability Assessment of Edge-Deployable LLMs for Location-Aware Tool-Use* > *(Accepted at IEEE ICWITE 2026)*

## Overview
The incentive towards sustainable, edge-deployable AI relies on small language models (≤3B parameters) that run on consumer hardware with low energy requirements. However, this repository demonstrates a critical safety-sustainability trade-off: under realistic input perturbations (typos, ambiguity), popular edge-scale models exhibit **severe semantic failure** in location extraction, while maintaining near-perfect JSON syntax.

This creates an **illusion of robustness**: production systems validating only JSON schema will certify incorrect outputs as correct, silently propagating hallucinated parameters to downstream tools.

## Key Findings

1. **The Semantic Cascade Rate (SCR):** We introduce SCR, a formal metric measuring the probability that a syntactically valid tool output contains a semantically incorrect parameter. SCR reaches **60–100%** under input perturbation.
2. **Silent Corruption vs. Hard Failure:** - **Qwen2.5-3B & Llama-3.2-3B:** Default to pre-training biases (e.g., "New York") when exact matching fails, producing valid but entirely incorrect JSON (*Silent Corruption*).
   - **Phi-3-mini:** Fails to produce valid JSON at all (*Hard Failure*), which is operationally safer as it can be caught by standard error-handling logic.
3. **Hardware Efficiency:** All evaluations were conducted on an NVIDIA RTX 3050 (6GB VRAM) using 4-bit NF4 quantization, costing approximately ~0.12 Wh per query.

### Performance Summary (Qwen2.5-3B)

| Input Condition | Syntactic Success (Valid JSON) | Semantic Success (Correct Location) | Semantic Cascade Rate (SCR) |
| :--- | :---: | :---: | :---: |
| **Clean** | 90% | 90% | 0% |
| **Typo (15% CER)** | 90% | 30% | 60% |
| **Ambiguity** | 100% | 0% | 100% |

##  Repository Structure

```text
├── data/
│   ├── clean_queries.json          # 10 base queries (e.g., "What is the weather in London?")
│   ├── typo_queries.json           # Queries with 15% KeyboardAug perturbation
│   └── ambiguity_queries.json      # Queries with location masked as "the city"
├── src/
│   ├── generate.py                 # Core inference script using 4-bit quantization
│   ├── evaluate_syntax.py          # jsonschema validation logic
│   ├── evaluate_semantics.py       # Regex entity matching & alias resolution
│   └── prompt_templates.py         # The 81-run prompt ablation configurations
├── notebooks/
│   └── Results_Analysis.ipynb      # Code to reproduce Table I, Table II, and Figures
├── requirements.txt
└── README.md
Getting Started
Prerequisites
You need a consumer-grade GPU with at least 6GB VRAM to run this evaluation locally (e.g., NVIDIA RTX 3050).

Installation
Clone the repository and install the required dependencies:

Bash
git clone [https://github.com/your-username/semantic-collapse-edge-ai.git](https://github.com/your-username/semantic-collapse-edge-ai.git)
cd semantic-collapse-edge-ai
pip install -r requirements.txt
Note: Ensure you have bitsandbytes installed for 4-bit NF4 quantization.

Running the Evaluation
To reproduce the core 2-step tool-use chain (Weather → Umbrella) across all perturbation conditions:

Bash
# Run evaluation for Qwen2.5-3B
python src/generate.py --model "Qwen/Qwen2.5-3B-Instruct" --quantization 4bit

# Run evaluation for Llama-3.2-3B
python src/generate.py --model "meta-llama/Llama-3.2-3B-Instruct" --quantization 4bit

# Calculate Syntactic and Semantic Success Rates
python src/evaluate_semantics.py --results_dir ./outputs/
Safety Recommendations for Practitioners
If you are deploying edge LLMs for agentic tool-use, we strongly recommend implementing a semantic validation layer. Before executing downstream tools (Step 2), perform a low-cost Regex-based entity check to verify that the extracted parameters actually appear in the original user query (Step 1).

Citation
If you find this code or research useful in your work, please cite our paper:

Code snippet
@inproceedings{anonymized2026brittle,
  title={Brittle Semantic Collapse Under Perturbation: A Safety and Sustainability Assessment of Edge-Deployable LLMs for Location-Aware Tool-Use},
  author={[Anonymized for Review]},
  booktitle={Proceedings of the IEEE International Conference on Wireless Information Technology and Edge Computing (ICWITE)},
  year={2026},
  organization={IEEE}
}
(Citation will be updated with author names upon de-anonymization).

License
This project is licensed under the MIT License - see the LICENSE file for details.


***

### Tips for Customizing:
1. **Badges:** Update the `[link-to-your-paper]` URL in the badges section once your paper is published or uploaded to ArXiv.
2. **Repository Structure:** I made highly educated guesses about how your python scripts might be named based on your methodology (`generate.py`, `evaluate_semantics.py`). If your files have different names, just tweak the file tree to match your actual code structure. 
3. **URL/Username:** Don't forget to replace `your-username` in the `git clone`
