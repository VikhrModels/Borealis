"""
Aggregate individual checkpoint evaluation results and find the best one.
"""
import json
from pathlib import Path

RESULTS_DIR = Path("/home/alex/Borealis")
CHECKPOINTS = ["checkpoint-1800", "checkpoint-2100", "checkpoint-2400", "checkpoint-2700", "checkpoint-2964"]

def main():
    all_results = {}

    for ckpt in CHECKPOINTS:
        result_file = RESULTS_DIR / f"eval_results_{ckpt}.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                all_results[ckpt] = {
                    "avg_wer": data["avg_wer"],
                    "splits": data["splits"]
                }
                print(f"{ckpt}: avg WER = {data['avg_wer']:.2f}%")
        else:
            print(f"{ckpt}: NOT FOUND")

    if not all_results:
        print("\nNo results found!")
        return

    # Find best checkpoint
    best_ckpt = min(all_results.items(), key=lambda x: x[1]["avg_wer"])
    print(f"\n{'='*60}")
    print(f"BEST CHECKPOINT: {best_ckpt[0]}")
    print(f"Average WER: {best_ckpt[1]['avg_wer']:.2f}%")
    print(f"{'='*60}")

    # Detailed results for best
    print(f"\nDetailed results for {best_ckpt[0]}:")
    for split, metrics in best_ckpt[1]["splits"].items():
        print(f"  {split}: WER={metrics['wer']:.2f}% CER={metrics['cer']:.2f}%")

    # Save aggregated results
    output = {
        "best_checkpoint": best_ckpt[0],
        "best_avg_wer": best_ckpt[1]["avg_wer"],
        "all_results": all_results
    }

    with open(RESULTS_DIR / "eval_results_aggregated.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved aggregated results to eval_results_aggregated.json")


if __name__ == "__main__":
    main()
