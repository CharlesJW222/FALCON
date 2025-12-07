#!/usr/bin/env python3
"""
Information Theory Analysis - Quick Start Script
"""

import argparse
import sys
import traceback
from pathlib import Path


def run_tofu_analysis(args):
    """
    TOFU dataset analysis
    """
    print("\n" + "="*80)
    print("TOFU INFORMATION THEORY ANALYSIS")
    print("="*80)
    
    try:
        from Falcon_github.MI2.Mutual_Info import UnifiedInformationAnalyzer
        
        analyzer = UnifiedInformationAnalyzer(
            model_name=args.model, 
            device="cuda" if args.device == "cuda" else "cpu"
        )
        
        original_estimate = analyzer.estimate_entropy
        def custom_estimate(activations, use_pca=True, n_components=args.pca_variance):
            return original_estimate(activations, use_pca, n_components)
        analyzer.estimate_entropy = custom_estimate
        
        if args.ratio:
            forget_data, retain_data = analyzer.load_tofu_data(
                forget_config=args.forget_config,
                retain_config=args.retain_config,
                sample_ratio=args.ratio
            )
        else:
            forget_data, retain_data = analyzer.load_tofu_data(
                forget_config=args.forget_config,
                retain_config=args.retain_config,
                max_samples=args.samples
            )
        
        print(f"\n{'='*60}")
        print("Sample from Forget Set:")
        print(f"{'='*60}")
        sample = forget_data[0]
        print(f"Q: {sample['question'][:150]}...")
        print(f"A: {sample['answer'][:150]}...")
        
        print(f"\nAnalyzing layer-wise mutual information...")
        layer_results = analyzer.analyze_layers(
            forget_data, 
            retain_data, 
            analyze_all=args.all_layers
        )
        
        if not layer_results:
            print("❌ No layer results obtained")
            return None
        
        min_layer, min_mi = analyzer.visualize_mutual_information(
            layer_results,
            save_path=args.output or "tofu_mi_analysis.png",
            dataset_name="TOFU"
        )
        
        summary = analyzer.print_analysis_summary(layer_results, "TOFU")
        
        return layer_results
        
    except Exception as e:
        print(f"❌ TOFU analysis failed: {e}")
        traceback.print_exc()
        return None


def run_wmdp_analysis(args):
    """
    WMDP dataset analysis
    """
    print("\n" + "="*80)
    print("WMDP INFORMATION THEORY ANALYSIS")
    print("="*80)
    
    try:
        from Falcon_github.MI2.Mutual_Info import UnifiedInformationAnalyzer
        
        analyzer = UnifiedInformationAnalyzer(
            model_name=args.model,
            device="cuda" if args.device == "cuda" else "cpu"
        )
        
        original_estimate = analyzer.estimate_entropy
        def custom_estimate(activations, use_pca=True, n_components=args.pca_variance):
            return original_estimate(activations, use_pca, n_components)
        analyzer.estimate_entropy = custom_estimate
        
        if args.ratio:
            forget_data, retain_data = analyzer.load_wmdp_data(
                forget_domain=args.domain,
                retain_dataset=args.retain,
                sample_ratio=args.ratio,
                max_length=args.max_length
            )
        else:
            forget_data, retain_data = analyzer.load_wmdp_data(
                forget_domain=args.domain,
                retain_dataset=args.retain,
                max_samples=args.samples,
                max_length=args.max_length
            )
        
        print(f"\n{'='*60}")
        print("Sample from Forget Set:")
        print(f"{'='*60}")
        sample = forget_data[0]
        text = sample['text'][:200] if 'text' in sample else str(sample)[:200]
        print(f"{text}...")
        
        print(f"\nAnalyzing layer-wise mutual information...")
        layer_results = analyzer.analyze_layers(
            forget_data,
            retain_data,
            analyze_all=args.all_layers
        )
        
        if not layer_results:
            print("❌ No layer results obtained")
            return None
        
        dataset_name = f"WMDP-{args.domain.upper()}"
        output_path = args.output or f"wmdp_{args.domain}_mi_analysis.png"
        
        min_layer, min_mi = analyzer.visualize_mutual_information(
            layer_results,
            save_path=output_path,
            dataset_name=dataset_name
        )
        
        summary = analyzer.print_analysis_summary(layer_results, dataset_name)
        
        return layer_results
        
    except Exception as e:
        print(f"❌ WMDP analysis failed: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Unified Information Theory Analysis for LLM Unlearning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=["tofu", "wmdp"],
        help="Dataset to analyze: 'tofu' or 'wmdp'"
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name (default: meta-llama/Llama-3.2-1B-Instruct)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Maximum samples to analyze (default: 30)"
    )
    
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Sample ratio (0.0-1.0), overrides --samples if specified"
    )
    
    parser.add_argument(
        "--forget-config",
        type=str,
        default="forget10",
        choices=["forget01", "forget05", "forget10"],
        help="TOFU forget configuration (default: forget10)"
    )
    
    parser.add_argument(
        "--retain-config",
        type=str,
        default="retain90",
        choices=["retain90", "retain95", "retain99"],
        help="TOFU retain configuration (default: retain90)"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        default="bio",
        choices=["bio", "cyber", "both"],
        help="WMDP forget domain: 'bio', 'cyber', or 'both' (default: bio)"
    )
    
    parser.add_argument(
        "--retain",
        type=str,
        default="wikitext",
        choices=["wikitext", "wmdp-corpus"],
        help="WMDP retain dataset (default: wikitext)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum text length for WMDP (default: 200)"
    )
    
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="PCA variance to retain (0.0-1.0, default: 0.95)"
    )
    
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Analyze all layers (slower but comprehensive)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for visualization (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    if args.ratio is not None and (args.ratio <= 0 or args.ratio > 1):
        print("❌ Error: --ratio must be between 0.0 and 1.0")
        return 1
    
    if args.pca_variance <= 0 or args.pca_variance > 1:
        print("❌ Error: --pca-variance must be between 0.0 and 1.0")
        return 1
    
    if args.samples <= 0:
        print("❌ Error: --samples must be positive")
        return 1
    
    print("\n" + "="*80)
    print("FALCON: Information Theory Analysis for LLM Unlearning")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset:          {args.dataset.upper()}")
    print(f"  Model:            {args.model}")
    print(f"  Device:           {args.device.upper()}")
    
    if args.ratio:
        print(f"  Sample ratio:     {args.ratio:.1%}")
    else:
        print(f"  Max samples:      {args.samples}")
    
    print(f"  PCA variance:     {args.pca_variance:.1%}")
    print(f"  Analyze all:      {'Yes' if args.all_layers else 'No (representative layers)'}")
    
    if args.dataset == "tofu":
        print(f"\nTOFU Settings:")
        print(f"  Forget config:    {args.forget_config}")
        print(f"  Retain config:    {args.retain_config}")
    
    elif args.dataset == "wmdp":
        print(f"\nWMDP Settings:")
        print(f"  Forget domain:    {args.domain}")
        print(f"  Retain dataset:   {args.retain}")
        print(f"  Max text length:  {args.max_length}")
    
    print("="*80)
    
    if args.dataset == "tofu":
        results = run_tofu_analysis(args)
    elif args.dataset == "wmdp":
        results = run_wmdp_analysis(args)
    else:
        print(f"❌ Unknown dataset: {args.dataset}")
        return 1
    
    if results is not None:
        print("\n" + "="*80)
        print("✓ Analysis completed successfully!")
        print("="*80)
        
        if args.output:
            print(f"✓ Results saved to: {args.output}")
        else:
            dataset_name = args.dataset
            if args.dataset == "wmdp":
                dataset_name = f"{args.dataset}_{args.domain}"
            print(f"✓ Results saved to: {dataset_name}_mi_analysis.png")
        return 0
    else:
        print("\n" + "="*80)
        print("❌ Analysis failed. Please check the error messages above.")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit(main())