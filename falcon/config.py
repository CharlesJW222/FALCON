"""
Configuration and data loading for unlearning task
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
from datasets import load_dataset

random.seed(0)


@dataclass
class UnlearningConfig:
    """Configuration for unlearning parameters"""
    model_path: str = "HuggingFaceH4/zephyr-7b-beta"
    module_str: str = "{model_name}.model.layers[{layer_id}]"
    output_dir: str = None
    
    retain_corpora: List[str] = field(default_factory=lambda: ["wikitext", "wikitext"])
    forget_corpora: List[str] = field(default_factory=lambda: ["bio-remove-dataset", "cyber-forget-corpus"])
    
    alpha: List[float] = field(default_factory=lambda: [100.0, 100.0])
    steering_coeffs: List[float] = field(default_factory=lambda: [20.0, 20.0])
     
    lr: float = 5e-5
    min_len: int = 0
    max_len: int = 2000
    batch_size: int = 4
    max_num_batches: int = 80
    conflict_weights: Tuple[float, float] = (0.8, 1.2) 
    align_weights: Tuple[float, float] = (0.1, 1.9)  
    
    layer_id: int = 7
    layer_ids: List[int] = field(default_factory=lambda: [5, 6, 7])
    param_ids: List[int] = field(default_factory=lambda: [6])
    
    seed: int = 42
    verbose: bool = False
    
    @staticmethod
    def from_args(args):
        """Create config from command line arguments"""
        return UnlearningConfig(
            model_path=args.model_name_or_path,
            module_str=args.module_str,
            output_dir=args.output_dir,
            retain_corpora=args.retain_corpora,
            forget_corpora=args.forget_corpora,
            alpha=args.alpha,
            steering_coeffs=args.steering_coeff_list,
            conflict_weights=args.conflict_weights,  
            align_weights=args.align_weights, 
            lr=args.lr,
            min_len=args.min_len,
            max_len=args.max_len,
            batch_size=args.batch_size,
            max_num_batches=args.max_num_batches,
            layer_id=args.layer_id,
            layer_ids=args.layer_ids,
            param_ids=args.param_ids,
            seed=args.seed,
            verbose=args.verbose
        )
    
    def print_config(self):
        """Print configuration"""
        print("====Unlearning Config====")
        for key, value in vars(self).items():
            print(f"{key}={value}")
        print("=====")


def load_corpus_data(corpus_name: str, min_len: int = 50, max_len: int = 2000):
    """Load data from a single corpus"""
    data = []
    
    if corpus_name == "wikitext":
        raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        for x in raw_data:
            if len(x['text']) > min_len:
                data.append(str(x['text']))
    else:
        data_path = f"/LOCAL2/sgjhu13/wmdp/data/{corpus_name}.jsonl"
        with open(data_path, "r") as f:
            for line in f:
                if "bio-remove-dataset" in corpus_name:
                    raw_text = json.loads(line)['text']
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
    
    return data


def prepare_batched_data(
    forget_corpora: List[str],
    retain_corpora: List[str],
    min_len: int = 50,
    max_len: int = 2000,
    batch_size: int = 4
):
    """Prepare batched data for training"""
    forget_data_list = []
    for corpus in forget_corpora:
        corpus_data = load_corpus_data(corpus, min_len, max_len)
        batches = [corpus_data[i:i + batch_size] for i in range(0, len(corpus_data), batch_size)]
        forget_data_list.append(batches)
    
    retain_data_list = []
    for corpus in retain_corpora:
        corpus_data = load_corpus_data(corpus, min_len, max_len)
        batches = [corpus_data[i:i + batch_size] for i in range(0, len(corpus_data), batch_size)]
        retain_data_list.append(batches)
    
    return forget_data_list, retain_data_list


def prepare_merged_data(
    forget_corpora: List[str],
    retain_corpora: List[str],
    min_len: int = 50,
    max_len: int = 2000,
    batch_size: int = 4
):
    """Prepare merged and shuffled data"""
    forget_data = []
    for corpus in forget_corpora:
        corpus_data = load_corpus_data(corpus, min_len, max_len)
        forget_data.extend(corpus_data)
    
    retain_data = []
    for corpus in retain_corpora:
        corpus_data = load_corpus_data(corpus, min_len, max_len)
        retain_data.extend(corpus_data)
    
    random.shuffle(forget_data)
    random.shuffle(retain_data)
    
    forget_batches = [forget_data[i:i + batch_size] for i in range(0, len(forget_data), batch_size)]
    retain_batches = [retain_data[i:i + batch_size] for i in range(0, len(retain_data), batch_size)]
    
    return [forget_batches], [retain_batches]
