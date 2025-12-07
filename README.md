# FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **"FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model"** accepted at **NeurIPS 2025**.

[📄 Paper](https://arxiv.org/abs/2502.01472) | [🌐 Project Page](#) | [📊 Datasets](#datasets)

---

## 🎯 Overview

FALCON is a novel representation-guided unlearning framework for Large Language Models (LLMs) that addresses the critical challenge of selectively removing undesired knowledge while preserving model utility. Unlike existing coarse-grained approaches, FALCON introduces:

- **🧭 Information-Theoretic Guidance**: Mutual information-based parameter selection for identifying layers with minimal knowledge entanglement
- **🎯 Contrastive Orthogonal Unalignment**: Fine-grained representation manipulation through Principal Offset Vectors (POVs) and gradient projection
- **🛡️ Enhanced Robustness**: Superior resistance against knowledge recovery attempts (e.g., jailbreaking attacks)

<div align="center">
  <img src="assets/falcon_overview.png" alt="FALCON Framework" width="800"/>
  <p><i>FALCON's three-stage pipeline: MI-guided parameter selection, contrastive representation unlearning, and orthogonal gradient conflict resolution.</i></p>
</div>

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YourUsername/FALCON.git
cd FALCON

# Create conda environment
conda create -n falcon python=3.9
conda activate falcon

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
zeta-learn>=0.1.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## 📊 Datasets

FALCON supports multiple unlearning benchmarks:

### 1. WMDP (Harmful Knowledge)
- **Bio-security**: Biological hazard knowledge
- **Cyber-security**: Cybersecurity threat knowledge
- Download: Follow instructions in `data/wmdp/README.md`

### 2. TOFU (Entity Unlearning)
- Fictitious author biographies
- Configurations: `forget01`, `forget05`, `forget10`

### 3. MUSE (Copyrighted Content)
- News articles and book excerpts
- Evaluation metrics: verbatim/knowledge memorization

---

## 🔧 Usage

### Step 1: Mutual Information Analysis

Identify optimal layers for unlearning using information-theoretic guidance:

```bash
# TOFU dataset analysis
python run_MI.py \
    --dataset tofu \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --forget-config forget10 \
    --retain-config retain90 \
    --samples 30 \
    --all-layers

# WMDP dataset analysis
python run_MI.py \
    --dataset wmdp \
    --model HuggingFaceH4/zephyr-7b-beta \
    --domain bio \
    --retain wikitext \
    --samples 30 \
    --output wmdp_bio_mi_heatmap.png
```

**Output**: Heatmap visualization showing mutual information across layers, identifying optimal intervention points.

### Step 2: Execute Unlearning

Run FALCON unlearning with optimized hyperparameters:

```bash
# WMDP harmful knowledge unlearning
python -m unlearning \
    --model_name_or_path HuggingFaceH4/zephyr-7b-beta \
    --forget_corpora bio-remove-dataset,cyber-forget-corpus \
    --retain_corpora wikitext,wikitext \
    --alpha 100,100 \
    --steering_coeffs 20,20 \
    --conflict_weights 0.8,1.2 \
    --align_weights 0.1,1.9 \
    --lr 5e-5 \
    --max_num_batches 80 \
    --layer_id 7 \
    --layer_ids 5,6,7 \
    --param_ids 6 \
    --output_dir ./outputs/zephyr_wmdp_unlearned

# TOFU entity unlearning
python -m unlearning \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --forget_corpora tofu_forget10 \
    --retain_corpora tofu_retain90 \
    --alpha 50 \
    --steering_coeffs 15 \
    --lr 1e-4 \
    --layer_id 3 \
    --output_dir ./outputs/llama_tofu_unlearned
```

### Step 3: Evaluation

Evaluate unlearning effectiveness and model utility:

```bash
# WMDP evaluation
python eval/evaluate_wmdp.py \
    --model_path ./outputs/zephyr_wmdp_unlearned \
    --dataset bio,cyber \
    --baseline_path HuggingFaceH4/zephyr-7b-beta

# TOFU evaluation
python eval/evaluate_tofu.py \
    --model_path ./outputs/llama_tofu_unlearned \
    --split forget10
```

---

## 📖 Interactive Tutorial

We provide a Jupyter notebook for hands-on exploration:

```bash
jupyter notebook quick_start.ipynb
```

The notebook covers:
1. Loading pre-trained models
2. MI-guided layer selection visualization
3. Step-by-step FALCON unlearning
4. Evaluation and result analysis

---

## 🏗️ Project Structure

```
FALCON/
├── algorithms.py           # Core algorithms (POV generation, contrastive loss)
├── config.py              # Configuration and data loading
├── model_tools.py         # Model utilities and activation extraction
├── unlearning.py          # Main training pipeline
├── Mutual_Info.py         # Information theory analysis
├── run_MI.py              # MI analysis script
├── quick_start.ipynb      # Interactive tutorial
├── eval/                  # Evaluation scripts
│   ├── evaluate_wmdp.py
│   ├── evaluate_tofu.py
│   └── evaluate_muse.py
├── data/                  # Dataset utilities
│   ├── wmdp/
│   ├── tofu/
│   └── muse/
├── assets/                # Figures and visualizations
└── requirements.txt
```

---

## 🔬 Key Components

### 1. Information-Theoretic Parameter Selection

```python
from Mutual_Info import UnifiedInformationAnalyzer

analyzer = UnifiedInformationAnalyzer(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    device="cuda"
)

# Analyze mutual information across layers
layer_results = analyzer.analyze_layers(
    forget_data, 
    retain_data, 
    analyze_all=True
)

# Identify optimal layer
min_layer, min_mi = analyzer.visualize_mutual_information(
    layer_results,
    save_path="mi_heatmap.png"
)
```

### 2. Contrastive Orthogonal Unalignment

```python
from algorithms import (
    generate_steering_vector,
    compute_contrastive_loss,
    resolve_gradient_conflict
)

# Generate Principal Offset Vector
steering_vec = generate_steering_vector(model, hidden_states)

# Compute contrastive loss
unlearn_loss = compute_contrastive_loss(
    anchor=train_activations,
    positive=steering_vec,
    negatives=frozen_activations,
    temperature=0.7
)

# Resolve gradient conflicts
final_grads, cos_sims = resolve_gradient_conflict(
    unlearn_grads,
    retain_grads,
    conflict_w=(0.8, 1.2),
    align_w=(0.1, 1.9)
)
```

### 3. Configuration System

```python
from config import UnlearningConfig

config = UnlearningConfig(
    model_path="HuggingFaceH4/zephyr-7b-beta",
    layer_id=7,                      # From MI analysis
    alpha=[100.0, 100.0],           # Retention weights
    steering_coeffs=[20.0, 20.0],   # POV scaling
    conflict_weights=(0.8, 1.2),    # Gradient conflict resolution
    align_weights=(0.1, 1.9),       # Gradient alignment weights
    lr=5e-5,
    max_num_batches=80
)
```

---

## 📈 Experimental Results

### WMDP Benchmark (Harmful Knowledge Unlearning)

| Method | WMDP-Bio ↓ | WMDP-Cyber ↓ | MMLU ↑ | PPL ↓ |
|--------|-----------|--------------|--------|-------|
| Zephyr-7B (Original) | 63.7 | 43.8 | 58.1 | 1.5 |
| LLMU | 36.3 | 40.5 | 50.3 | 4.8 |
| SCRUB | 38.7 | 35.4 | 50.0 | 16.5 |
| RMU | 34.5 | 28.9 | 57.4 | 1.5 |
| **FALCON** | **26.7** | **25.3** | **57.4** | **1.5** |

### TOFU Benchmark (Entity Unlearning)

| Method | Forget Quality ↑ | Model Utility ↑ |
|--------|-----------------|----------------|
| Finetuned | 0.01 | 0.60 |
| GradAscent | 0.27 | 0.33 |
| NPO | 0.92 | 0.56 |
| RMU | 0.16 | 0.55 |
| **FALCON** | **0.99** | **0.55** |

### Knowledge Recovery Resistance

| Attack | Original | RMU | FALCON |
|--------|----------|-----|---------|
| GCG-500 | 65.4 | 58.5±3.2 | **28.1±0.5** |
| GCG-2000 | 65.4 | 60.2±2.8 | **28.9±0.8** |

---

## 🛠️ Advanced Configuration

### Custom Hyperparameters

```bash
python -m unlearning \
    --model_name_or_path YOUR_MODEL \
    --conflict_weights 0.5,1.5 \      # Custom conflict resolution
    --align_weights 0.2,1.8 \         # Custom alignment weights
    --steering_coeffs 25,30 \         # Domain-specific POV scaling
    --alpha 150,120 \                 # Asymmetric retention weights
    --lr 1e-4 \
    --max_num_batches 100
```

### Multi-Domain Unlearning

```bash
# Unlearn multiple knowledge domains
python -m unlearning \
    --forget_corpora bio-dataset,cyber-dataset,toxic-dataset \
    --retain_corpora wikitext,wikitext,wikitext \
    --alpha 100,100,80 \              # Per-domain retention weights
    --steering_coeffs 20,20,15        # Per-domain POV scaling
```

---

## 📝 Citation

If you find FALCON useful in your research, please cite:

```bibtex
@inproceedings{hu2025falcon,
  title={FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model},
  author={Hu, Jinwei and Huang, Zhenglin and Yin, Xiangyu and Ruan, Wenjie and Cheng, Guangliang and Dong, Yi and Huang, Xiaowei},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions and discussions:

- **Jinwei Hu**: jinwei.hu@liverpool.ac.uk
- **Yi Dong**: yi.dong@liverpool.ac.uk (Corresponding Author)
- **Xiaowei Huang**: xiaowei.huang@liverpool.ac.uk (Corresponding Author)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- This work is partially funded by the European Union (Grant Agreement ID 101212818)
- Partially supported by Innovate UK through AI-PASSPORT (Grant 10126404)
- Special thanks to the open-source community for foundational tools and datasets

---

## ⚠️ Responsible AI Statement

FALCON is designed for responsible AI deployment. While our method enables selective knowledge removal from LLMs, users must ensure:

- Compliance with applicable regulations (e.g., GDPR's "right to be forgotten")
- Ethical considerations in determining what knowledge should be unlearned
- Transparency in communicating model capabilities and limitations
- Ongoing monitoring for unintended consequences

The authors do not endorse using this technology for malicious purposes or circumventing legitimate safety mechanisms.

---

## 🔗 Related Projects

- [RMU: Representation Misdirection for Unlearning](https://github.com/locuslab/rmu)
- [TOFU: Task of Fictitious Unlearning](https://github.com/locuslab/tofu)
- [WMDP: Weapons of Mass Destruction Proxy](https://github.com/centerforaisafety/wmdp)

---

<div align="center">
  <p>⭐ Star us on GitHub if you find FALCON useful!</p>
  <p>Made with ❤️ by the University of Liverpool AI Safety Team</p>
</div>
