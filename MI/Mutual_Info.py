import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class UnifiedInformationAnalyzer:
    """
    MI-based Information Analyzer for LLMs
    """
    
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        print("Model loaded successfully!")
        
    
    def load_tofu_data(self, forget_config="forget10", retain_config="retain90", 
                       max_samples=200, sample_ratio=None):
        """
        load TOFU dataset
        
        Args:
            forget_config: (forget01/forget05/forget10)
            retain_config: (retain90/retain95/retain99)
        
        Returns:
            forget_data, retain_data
        """
        print(f"\n{'='*60}")
        print(f"Loading TOFU Dataset")
        print(f"{'='*60}")
        print(f"Forget config: {forget_config}")
        print(f"Retain config: {retain_config}")
        
        forget_dataset = load_dataset("locuslab/TOFU", forget_config, split="train")
        retain_dataset = load_dataset("locuslab/TOFU", retain_config, split="train")
        
        print(f"\nOriginal dataset sizes:")
        print(f"  - {forget_config}: {len(forget_dataset)} samples")
        print(f"  - {retain_config}: {len(retain_dataset)} samples")
        
        if sample_ratio is not None:
            forget_samples = max(1, int(len(forget_dataset) * sample_ratio))
            retain_samples = max(1, int(len(retain_dataset) * sample_ratio))
            print(f"\nUsing {sample_ratio:.1%} of data:")
        else:
            forget_samples = min(max_samples, len(forget_dataset))
            retain_samples = min(max_samples, len(retain_dataset))
            print(f"\nUsing max {max_samples} samples:")
        
        forget_data = forget_dataset.select(range(forget_samples))
        retain_data = retain_dataset.select(range(retain_samples))
        
        print(f"  - Forget set: {len(forget_data)} samples ({len(forget_data)/len(forget_dataset):.1%})")
        print(f"  - Retain set: {len(retain_data)} samples ({len(retain_data)/len(retain_dataset):.1%})")
        
        return forget_data, retain_data
    
    def load_wmdp_data(self, forget_domain="bio", retain_dataset="wikitext", 
                       max_samples=200, sample_ratio=None, max_length=200):
        """
        load WMDP dataset
        
        Args:
            forget_domain: "bio", "cyber", or "both"
            retain_dataset: "wikitext" or "wmdp-corpus"
            max_samples: 最大样本数量
            sample_ratio: 0.0-1.0
            max_length: max token length for each sample
        
        Returns:
            forget_data, retain_data (list of dicts)
        """
        print(f"\n{'='*60}")
        print(f"Loading WMDP Dataset")
        print(f"{'='*60}")
        print(f"Forget domain: {forget_domain}")
        print(f"Retain dataset: {retain_dataset}")
        
        forget_data = []
        
        if forget_domain in ["bio", "both"]:
            print("\nLoading WMDP-Bio...")
            try:
                bio_dataset = load_dataset("cais/wmdp", "wmdp-bio", split="test")
                print(f"  - WMDP-Bio loaded: {len(bio_dataset)} samples")
                forget_data.extend(self._process_wmdp_dataset(bio_dataset, "bio", max_length))
            except Exception as e:
                print(f"  - Warning: Failed to load WMDP-Bio: {e}")
        
        if forget_domain in ["cyber", "both"]:
            print("\nLoading WMDP-Cyber...")
            try:
                cyber_dataset = load_dataset("cais/wmdp", "wmdp-cyber", split="test")
                print(f"  - WMDP-Cyber loaded: {len(cyber_dataset)} samples")
                forget_data.extend(self._process_wmdp_dataset(cyber_dataset, "cyber", max_length))
            except Exception as e:
                print(f"  - Warning: Failed to load WMDP-Cyber: {e}")
        
        print(f"\nLoading retain dataset: {retain_dataset}...")
        retain_data = []
        
        if retain_dataset == "wikitext":
            try:
                wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                print(f"  - WikiText loaded: {len(wiki_dataset)} samples")
                retain_data = self._process_wikitext(wiki_dataset, max_length)
            except Exception as e:
                print(f"  - Warning: Failed to load WikiText: {e}")
        
        elif retain_dataset == "wmdp-corpus":
            try:
                corpus_dataset = load_dataset("cais/wmdp-corpora", split="train")
                print(f"  - WMDP-Corpus loaded: {len(corpus_dataset)} samples")
                retain_data = self._process_wmdp_corpus(corpus_dataset, max_length)
            except Exception as e:
                print(f"  - Warning: Failed to load WMDP-Corpus: {e}")
        
        print(f"\nOriginal sizes:")
        print(f"  - Forget set: {len(forget_data)} samples")
        print(f"  - Retain set: {len(retain_data)} samples")
        
        if sample_ratio is not None:
            forget_samples = max(1, int(len(forget_data) * sample_ratio))
            retain_samples = max(1, int(len(retain_data) * sample_ratio))
            print(f"\nUsing {sample_ratio:.1%} of data:")
        else:
            forget_samples = min(max_samples, len(forget_data))
            retain_samples = min(max_samples, len(retain_data))
            print(f"\nUsing max {max_samples} samples:")
        
        forget_data = forget_data[:forget_samples]
        retain_data = retain_data[:retain_samples]
        
        print(f"  - Forget set: {len(forget_data)} samples")
        print(f"  - Retain set: {len(retain_data)} samples")
        
        return forget_data, retain_data
    
    def _process_wmdp_dataset(self, dataset, domain, max_length=200):
        """
        WMDP preprocessing
        
        WMDP format example:
        {
            'question': 'What is...?',
            'choices': ['A. ...', 'B. ...', 'C. ...', 'D. ...'],
            'answer': 2  
        }
        """
        processed_data = []
        
        for item in dataset:
            question = item['question']
            choices = item.get('choices', [])
            
            if choices:
                text = f"Question: {question}\n"
                for i, choice in enumerate(choices):
                    text += f"{chr(65+i)}. {choice}\n"
                text = text.strip()
            else:
                text = f"Question: {question}"
            
            if len(text) > max_length:
                text = text[:max_length]
            
            processed_data.append({
                'text': text,
                'domain': domain,
                'question': question
            })
        
        return processed_data
    
    def _process_wikitext(self, dataset, max_length=200):
        processed_data = []
        
        for item in dataset:
            text = item['text'].strip()
            
            if not text or text.startswith('=') or len(text) < 50:
                continue
            
            if len(text) > max_length:
                text = text[:max_length]
            
            processed_data.append({
                'text': text,
                'domain': 'wikitext'
            })
        
        return processed_data
    
    def _process_wmdp_corpus(self, dataset, max_length=200):
        processed_data = []
        
        for item in dataset:
            text = item.get('text', '').strip()
            
            if not text or len(text) < 50:
                continue
            
            if len(text) > max_length:
                text = text[:max_length]
            
            processed_data.append({
                'text': text,
                'domain': item.get('domain', 'corpus')
            })
        
        return processed_data
    
    
    def prepare_input_text(self, sample):
        """
        unify input text preparation for different datasets
        """
        if isinstance(sample, str):
            return sample
        
        if 'question' in sample and 'answer' in sample:
            question = sample['question']
            answer = sample['answer']
            if answer:
                text = f"Question: {question}\nAnswer: {answer}"
            else:
                text = f"Question: {question}\nAnswer:"
            return text
        
        if 'text' in sample:
            return sample['text']
        
        return str(sample)
    
    
    def extract_activations(self, dataset, layer_indices=None, max_length=512):
        """
        extract activations from specified layers
        
        Args:
            dataset: list or HuggingFace Dataset
            layer_indices: indices of layers to extract activations from
            max_length: maximum length for inputs
        
        Returns:
            dict: {layer_idx: np.array of activations}
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.model.model.layers)))
        
        all_activations = {i: [] for i in layer_indices}
        
        print(f"Extracting activations from {len(dataset)} samples...")
        
        with torch.no_grad():
            for sample in tqdm(dataset):
                text = self.prepare_input_text(sample)
                
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length
                ).to(self.device)
                
                activations = {}
                
                def make_hook(layer_idx):
                    def hook(module, input, output):
                        activations[layer_idx] = output[0][:, -1, :].cpu().numpy()
                    return hook
                
                hooks = []
                for layer_idx in layer_indices:
                    hook = self.model.model.layers[layer_idx].register_forward_hook(
                        make_hook(layer_idx)
                    )
                    hooks.append(hook)
                
                _ = self.model(**inputs)
                
                for layer_idx in layer_indices:
                    if layer_idx in activations:
                        all_activations[layer_idx].append(activations[layer_idx])
                
                for hook in hooks:
                    hook.remove()
        
        for layer_idx in layer_indices:
            if all_activations[layer_idx]:
                all_activations[layer_idx] = np.vstack(all_activations[layer_idx])
            
        return all_activations
    
    
    def estimate_entropy(self, activations, use_pca=True, n_components=0.95):
        """
        used to estimate the entropy of activations
        
        Args:
            activations:  (n_samples, n_features)
        
        Returns:
            float: estimated entropy H(X)
        """
        if len(activations) == 0:
            return 0.0
            
        n_samples, n_features = activations.shape
        
        if not use_pca or n_features <= 10:
            print(f"  Using raw activations: {activations.shape}")
            activations_reduced = activations
        else:
            try:
                if isinstance(n_components, float) and 0 < n_components < 1:
                    print(f"  Using PCA to retain {n_components:.1%} variance...")
                    pca = PCA(n_components=n_components)
                    activations_reduced = pca.fit_transform(activations)
                    actual_components = pca.n_components_
                    variance_retained = pca.explained_variance_ratio_.sum()
                    print(f"  PCA: {n_features}→{actual_components} dims, "
                          f"variance retained: {variance_retained:.3f}")
                
                else:
                    max_components = min(n_samples - 1, n_features)
                    if max_components <= 0:
                        print(f"  Warning: Too few samples ({n_samples}), using raw activations")
                        activations_reduced = activations
                    else:
                        actual_components = min(int(n_components), max_components)
                        if actual_components != n_components:
                            print(f"  Adjusting components: {n_components}→{actual_components}")
                        
                        pca = PCA(n_components=actual_components)
                        activations_reduced = pca.fit_transform(activations)
                        variance_retained = pca.explained_variance_ratio_.sum()
                        print(f"  PCA: {n_features}→{actual_components} dims, "
                              f"variance retained: {variance_retained:.3f}")
                        
            except Exception as e:
                print(f"  PCA failed ({e}), using raw activations")
                activations_reduced = activations
        
        try:
            kde = gaussian_kde(activations_reduced.T)
            
            log_densities = kde.logpdf(activations_reduced.T)
            
            entropy = -np.mean(log_densities)
            
            return entropy
        
        except Exception as e:
            print(f"  Error estimating entropy: {e}")
            return 0.0
    
    def estimate_joint_entropy(self, activations1, activations2):
        """ 
        Args:
            activations1: forget data
            activations2: retain data
        
        Returns:
            float: joint entropy H(X,Y)
        """
        if len(activations1) == 0 or len(activations2) == 0:
            return 0.0

        min_samples = min(len(activations1), len(activations2))
        activations1 = activations1[:min_samples]
        activations2 = activations2[:min_samples]
        
        joint_activations = np.hstack([activations1, activations2])
        
        return self.estimate_entropy(joint_activations)
    
    
    def calculate_mutual_information(self, forget_activations, retain_activations):
        if len(forget_activations) == 0 or len(retain_activations) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        print("  Computing entropies...")
        h_forget = self.estimate_entropy(forget_activations)
        h_retain = self.estimate_entropy(retain_activations)
        h_joint = self.estimate_joint_entropy(forget_activations, retain_activations)
        
        mutual_info = h_forget + h_retain - h_joint
        
        return mutual_info, h_forget, h_retain, h_joint
    
    
    def analyze_layers(self, forget_data, retain_data, sample_layers=None, analyze_all=False):
        """
        l* = argmin_l I(F^(l); R^(l))
        
        Args:
            forget_data
            retain_data
            sample_layers
            analyze_al
        
        Returns:
            dict: {layer_idx: {'mutual_information': ..., ...}}
        """
        total_layers = len(self.model.model.layers)
        
        if analyze_all:
            analysis_layers = list(range(total_layers))
            print(f"\n{'='*60}")
            print(f"Analyzing ALL {total_layers} layers (this may take a while...)")
            print(f"{'='*60}")
        
        elif sample_layers is not None:
            analysis_layers = sample_layers
            print(f"\n{'='*60}")
            print(f"Analyzing specified layers: {analysis_layers}")
            print(f"{'='*60}")
        
        else:
            analysis_layers = list(range(0, total_layers, max(1, total_layers // 12)))
            print(f"\n{'='*60}")
            print(f"Analyzing sampled layers: {analysis_layers}")
            print(f"(Out of {total_layers} total layers)")
            print(f"{'='*60}")
        
        # extract activations
        forget_activations = self.extract_activations(forget_data, analysis_layers)
        retain_activations = self.extract_activations(retain_data, analysis_layers)
        
        # calculate MI
        layer_results = {}
        
        for i, layer_idx in enumerate(analysis_layers):
            print(f"\nAnalyzing Layer {layer_idx} ({i+1}/{len(analysis_layers)})...")
            
            if (layer_idx in forget_activations and layer_idx in retain_activations and
                len(forget_activations[layer_idx]) > 0 and len(retain_activations[layer_idx]) > 0):
                
                mi, h_f, h_r, h_joint = self.calculate_mutual_information(
                    forget_activations[layer_idx], 
                    retain_activations[layer_idx]
                )
                
                layer_results[layer_idx] = {
                    'mutual_information': mi,
                    'forget_entropy': h_f,
                    'retain_entropy': h_r,
                    'joint_entropy': h_joint
                }
                
                print(f"  ✓ Layer {layer_idx}: MI = {mi:.4f}, "
                      f"H(F) = {h_f:.4f}, H(R) = {h_r:.4f}, H(F,R) = {h_joint:.4f}")
            else:
                print(f"  ✗ Layer {layer_idx}: Skipped (no valid activations)")
        
        return layer_results
    
    
    def visualize_mutual_information(self, layer_results, save_path=None, 
                                    dataset_name="Dataset"):
        layers = sorted(layer_results.keys())
        mi_values = [layer_results[layer]['mutual_information'] for layer in layers]
        
        # normalization
        min_mi, max_mi = np.min(mi_values), np.max(mi_values)
        mi_range = max_mi - min_mi
        
        if mi_range > 0:
            normalized_mi = [(mi - min_mi) / mi_range for mi in mi_values]
        else:
            normalized_mi = [0.0] * len(mi_values)
        
        plt.figure(figsize=(18, 10))
        
        plt.subplot(1, 2, 1)
        norm_matrix = np.array(normalized_mi).reshape(1, -1)
        sns.heatmap(
            norm_matrix, 
            xticklabels=[f'L{i}' for i in layers],
            yticklabels=['Normalized MI'],
            annot=True, 
            fmt='.3f', 
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Normalized MI (0=Best, 1=Worst)'},
            vmin=0, vmax=1
        )
        plt.title(f'{dataset_name}: Normalized Mutual Information Heatmap', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        
        colors = ['green' if x <= 0.1 else 'orange' if x <= 0.6 else 'red' 
                  for x in normalized_mi]
        
        plt.plot(layers, normalized_mi, '-', linewidth=2, color='purple', alpha=0.5)
        
        plt.scatter(layers, normalized_mi, c=colors, s=150, alpha=0.7, 
                   edgecolors='black', linewidths=2, zorder=3)
        
        min_mi_layer = layers[np.argmin(mi_values)]
        min_mi_idx = np.argmin(mi_values)
        plt.scatter([min_mi_layer], [normalized_mi[min_mi_idx]], 
                   s=300, marker='*', c='gold', edgecolors='black', 
                   linewidths=2, zorder=4, label=f'Optimal: Layer {min_mi_layer}')
        
        plt.xlabel('Layer Index', fontsize=12, fontweight='bold')
        plt.ylabel('Normalized MI (0=Best, 1=Worst)', fontsize=12, fontweight='bold')
        plt.title(f'{dataset_name}: Layer-wise MI with Quality Levels', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
        
        plt.axhspan(0, 0.1, alpha=0.1, color='green', label='Excellent (≤0.1)')
        plt.axhspan(0.1, 0.6, alpha=0.1, color='orange', label='Good-Fair (0.1-0.6)')
        plt.axhspan(0.6, 1.0, alpha=0.1, color='red', label='Poor (>0.6)')
        
        plt.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {save_path}")
        
        plt.show()
        
        min_mi_value = min(mi_values)
        return min_mi_layer, min_mi_value
    
    def print_analysis_summary(self, layer_results, dataset_name="Dataset"):
        layers = sorted(layer_results.keys())
        mi_values = [layer_results[layer]['mutual_information'] for layer in layers]
        
        print("\n" + "="*80)
        print(f"INFORMATION THEORY ANALYSIS SUMMARY - {dataset_name}")
        print("="*80)
        
        print(f"\nModel Configuration:")
        print(f"  Model: {self.model.config._name_or_path}")
        print(f"  Total layers: {len(self.model.model.layers)}")
        print(f"  Analyzed layers: {len(layers)}")
        print(f"  Layer range: {min(layers)} - {max(layers)}")
        
        
        # normalization
        min_mi = np.min(mi_values)
        max_mi = np.max(mi_values)
        mi_range = max_mi - min_mi
        
        if mi_range > 0:
            normalized_mi = [(mi - min_mi) / mi_range for mi in mi_values]
        else:
            normalized_mi = [0.0] * len(mi_values)
        
        print("\n" + "="*80)
        print("DETAILED LAYER ANALYSIS")
        print("-" * 80)
        print(f"{'Layer':>5} |  {'Normalized':>11} |  {'Quality':>12}")
        print("-" * 80)
        
        for i, layer in enumerate(layers):
            result = layer_results[layer]
            norm_score = normalized_mi[i]
            
            if norm_score <= 0.1:
                level = "🟢 Excellent"
            elif norm_score <= 0.4:
                level = "🟡 Good"
            elif norm_score <= 0.6:
                level = "🟠 Fair"
            elif norm_score <= 0.8:
                level = "🔴 Poor"
            else:
                level = "⚫ Very Poor"
            
            print(f"{layer:5d} | "
                  f"{norm_score:11.4f} | "
                  f"{level}")
        
        print("-" * 80)
        
        best_layers = [layers[i] for i, norm in enumerate(normalized_mi) if norm <= 0.1]
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS FOR UNLEARNING")
        print("="*80)
        
        if best_layers:
            print(f"\n🟢 Unlearning candidates (normalized ≤ 0.1):")
            print(f"   Layers: {best_layers}")
            print(f"   Recommendation: Start with Layer {best_layers[0]}")
        
        optimal_layer = layers[np.argmin(mi_values)]
        print(f"\n⭐ OPTIMAL layer (lowest MI): Layer {optimal_layer}")

        
        print("\n" + "="*80)
        
        return {
            'original_mi': mi_values,
            'normalized_mi': normalized_mi,
            'layers': layers,
            'optimal_layer': optimal_layer,
            'best_layers': best_layers,
            'stats': {
                'min_mi': min_mi,
                'max_mi': max_mi,
                'mean_mi': np.mean(mi_values),
                'std_mi': np.std(mi_values),
                'range_mi': mi_range
            }
        }


