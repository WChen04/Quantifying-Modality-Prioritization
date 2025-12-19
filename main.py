import os
import time
import pandas as pd
import yaml
from tqdm import tqdm
from src.generator import ArtifactGenerator
from src.judge import SafetyJudge
from src.models import TargetModel


def load_config():
    """Load config and resolve environment variables."""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Run: export GEMINI_API_KEY='your-gemini-api-key-here'"
        )
    config["gemini_api_key"] = api_key
    return config


def run_experiment(mode: str = "omni"):
    """
    Run the Harmonic-Dissonance benchmark.
    
    Args:
        mode: Testing mode - 'text', 'visual', or 'omni' (audio+visual)
    """
    config = load_config()
    request_delay = config.get("request_delay", 5)
    
    # Initialize components
    artifacts_dir = config.get("artifacts_dir", "results/artifacts")
    generator = ArtifactGenerator(output_dir=artifacts_dir)
    judge = SafetyJudge()  # Keyword-based, no API needed
    target_model = TargetModel(
        api_key=config["gemini_api_key"], 
        model=config.get("model_target", "gemini-2.0-flash")
    )
    
    # Load dataset from CSV
    dataset = pd.read_csv("data/dataset.csv")
    print(f"[*] Loaded {len(dataset)} test cases from data/dataset.csv")
    
    results = []
    
    print(f"[*] Starting Harmonic-Dissonance Benchmark (mode: {mode})...")
    print(f"[*] Model: {config.get('model_target', 'gemini-2.5-flash-preview-04-17')}")
    print(f"[*] Evaluation: Keyword-based heuristic (Refusal/Compliance)")
    
    for _, item in tqdm(dataset.iterrows(), total=len(dataset)):
        item_id = str(item["id"]).zfill(3)
        
        # 1. Generate Attack Vectors
        img_path = generator.create_visual_threat(item["malicious"], item_id)
        audio_path = generator.create_audio_mask(item["benign_audio"], item_id)
        
        # 2. Run Inference based on mode
        if mode == "text":
            # Text-only baseline (direct malicious query)
            response_text = target_model.infer_text_only(item["malicious"])
        elif mode == "visual":
            # Visual-only (typographic attack without audio mask)
            response_text = target_model.infer_visual_only(img_path)
        else:
            # Omni-modal (visual + audio mask)
            response_text = target_model.infer(img_path, audio_path)
        
        # 3. Keyword-based Evaluation (no API call needed)
        eval_result = judge.evaluate(item["malicious"], response_text)
        
        # 4. Log Result
        results.append({
            "id": item_id,
            "category": item["category"],
            "mode": mode,
            "classification": eval_result.get("classification", "Error"),
            "score": eval_result.get("score", -1),
            "reasoning": eval_result.get("reasoning", ""),
            "response_snippet": response_text[:300] if response_text else ""
        })
        
        # Delay before next API call
        time.sleep(request_delay)
    
    # Save results
    os.makedirs(config.get("output_dir", "results"), exist_ok=True)
    output_file = f"results/experiment_{mode}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n[*] Experiment Complete ({mode} mode)")
    print(f"    Results saved to: {output_file}")
    
    # Calculate Attack Success Rate (ASR)
    valid_results = df[df["score"] >= 0]
    if len(valid_results) > 0:
        compliance_count = (valid_results["score"] == 1).sum()
        refusal_count = (valid_results["score"] == 0).sum()
        asr = compliance_count / len(valid_results) * 100
        
        print(f"\n    Classification Results:")
        print(f"    - Refusal:    {refusal_count} ({refusal_count/len(valid_results)*100:.1f}%)")
        print(f"    - Compliance: {compliance_count} ({compliance_count/len(valid_results)*100:.1f}%)")
        print(f"\n    Attack Success Rate (ASR): {asr:.1f}%")
    else:
        print("\n    No valid results to analyze")
    
    return df


def run_full_comparison():
    """Run all three modes for A/B comparison."""
    print("=" * 60)
    print("HARMONIC-DISSONANCE BENCHMARK - Full Comparison")
    print("=" * 60)
    
    results = {}
    for mode in ["text", "visual", "omni"]:
        print(f"\n{'='*60}")
        print(f"Running {mode.upper()} mode...")
        print("=" * 60)
        results[mode] = run_experiment(mode)
    
    # Compute Audio Multiplier Effect
    print("\n" + "=" * 60)
    print("AUDIO MULTIPLIER EFFECT ANALYSIS")
    print("=" * 60)
    
    # Calculate ASR for each mode
    asr_results = {}
    for mode, df in results.items():
        valid = df[df["score"] >= 0]
        if len(valid) > 0:
            asr_results[mode] = (valid["score"] == 1).mean() * 100
        else:
            asr_results[mode] = 0
    
    print(f"\nAttack Success Rates:")
    print(f"    Text-Only:   {asr_results['text']:.1f}%")
    print(f"    Visual-Only: {asr_results['visual']:.1f}%")
    print(f"    Omni-Modal:  {asr_results['omni']:.1f}%")
    
    if asr_results["visual"] > 0:
        multiplier = asr_results["omni"] / asr_results["visual"]
        print(f"\n    Audio Multiplier Effect: {multiplier:.2f}x")
    else:
        if asr_results["omni"] > 0:
            print(f"\n    Audio Multiplier Effect: ∞ (visual baseline is 0%)")
        else:
            print(f"\n    Audio Multiplier Effect: N/A (both are 0%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Harmonic-Dissonance Benchmark")
    parser.add_argument(
        "--mode", 
        choices=["text", "visual", "omni", "full"],
        default="omni",
        help="Experiment mode: text, visual, omni, or full (runs all three)"
    )
    args = parser.parse_args()
    
    if args.mode == "full":
        run_full_comparison()
    else:
        run_experiment(args.mode)
