import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from calibrate import BPETokenizer, Token, TokenSensitivityClassifier

try:
    from datasets import load_dataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


STAGES = ("regex", "regex+dict", "full")


@dataclass
class EvalCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0


def token_overlaps_span(tok: Token, start: int, end: int) -> bool:
    return tok.char_start < end and tok.char_end > start


def gold_token_labels(tokens: List[Token], privacy_mask: List[Dict]) -> Tuple[set[int], Dict[int, str]]:
    gold_positive: set[int] = set()
    gold_label_by_idx: Dict[int, str] = {}

    for span in privacy_mask:
        start = int(span["start"])
        end = int(span["end"])
        label = str(span["label"])
        for tok in tokens:
            if token_overlaps_span(tok, start, end):
                gold_positive.add(tok.idx)
                gold_label_by_idx.setdefault(tok.idx, label)

    return gold_positive, gold_label_by_idx


def compute_metrics(counts: EvalCounts) -> Dict[str, float]:
    tp, fp, tn, fn = counts.tp, counts.fp, counts.tn, counts.fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr,
    }


def iter_ai4privacy(split: str, limit: int | None, language: str | None) -> Iterable[Dict]:
    if not _DATASETS_AVAILABLE:
        raise RuntimeError("datasets is not installed; cannot load ai4privacy/pii-masking-300k")

    ds = load_dataset("ai4privacy/pii-masking-300k", split=split)
    if language:
        ds = ds.filter(lambda row: str(row.get("language", "")).lower() == language.lower())
    else:
        ds = ds.filter(lambda row: str(row.get("language", "")).lower() in ("english", "spanish"))
    ds = ds.shuffle(seed=42)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def evaluate_stage(
    rows: List[Dict],
    tokenizer: BPETokenizer,
    stage: str,
    text_limit_chars: int,
    cache_file: str | None,
) -> Dict:
    classifier = TokenSensitivityClassifier(cache_file=cache_file, stage=stage)
    total_counts = EvalCounts()
    by_language: Dict[str, EvalCounts] = defaultdict(EvalCounts)
    by_label_total: Counter = Counter()
    by_label_tp: Counter = Counter()
    n_examples = 0
    n_tokens = 0

    row_iter = rows
    if _TQDM_AVAILABLE:
        row_iter = tqdm(rows, desc=f"{stage} stage", unit="row")

    for row in row_iter:
        text = str(row.get("source_text", ""))
        if not text.strip():
            continue

        text = text[:text_limit_chars]
        privacy_mask = [
            span for span in row.get("privacy_mask", [])
            if int(span["start"]) < len(text)
        ]

        tokens = tokenizer.tokenize(text)
        predicted_sensitive = classifier.classify(tokens, text, language=str(row.get("language", "")))
        predicted_positive = {st.token.idx for st in predicted_sensitive}
        gold_positive, gold_label_by_idx = gold_token_labels(tokens, privacy_mask)

        counts = total_counts
        lang_counts = by_language[str(row.get("language", "unknown"))]

        for tok in tokens:
            pred = tok.idx in predicted_positive
            gold = tok.idx in gold_positive
            if pred and gold:
                counts.tp += 1
                lang_counts.tp += 1
                by_label_tp[gold_label_by_idx[tok.idx]] += 1
            elif pred and not gold:
                counts.fp += 1
                lang_counts.fp += 1
            elif not pred and gold:
                counts.fn += 1
                lang_counts.fn += 1
            else:
                counts.tn += 1
                lang_counts.tn += 1

            if gold:
                by_label_total[gold_label_by_idx[tok.idx]] += 1

        n_examples += 1
        n_tokens += len(tokens)

    classifier.save_cache()
    return {
        "stage": stage,
        "counts": total_counts,
        "metrics": compute_metrics(total_counts),
        "by_language": by_language,
        "by_label_total": by_label_total,
        "by_label_tp": by_label_tp,
        "n_examples": n_examples,
        "n_tokens": n_tokens,
        "detector_stats": classifier.stats,
    }


def print_summary(result: Dict) -> None:
    counts = result["counts"]
    metrics = result["metrics"]
    by_language = result["by_language"]
    by_label_total = result["by_label_total"]
    by_label_tp = result["by_label_tp"]

    print(f"\nToken-Level Sensitivity Detection Accuracy [{result['stage']}]")
    print("=" * 60)
    print(f"Examples evaluated: {result['n_examples']}")
    print(f"Tokens evaluated:   {result['n_tokens']}")
    print(f"TP={counts.tp} FP={counts.fp} TN={counts.tn} FN={counts.fn}")
    print()
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"FPR:       {metrics['fpr']:.4f}")
    print(f"FNR:       {metrics['fnr']:.4f}")

    print("\nDetector stats")
    print("-" * 60)
    print(
        f"cache_hits={result['detector_stats']['cache_hits']} "
        f"cache_misses={result['detector_stats']['cache_misses']} "
        f"nlp_calls={result['detector_stats']['nlp_calls']}"
    )

    if by_language:
        print("\nBy language")
        print("-" * 60)
        print(f"{'Language':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support'}")
        for language, lang_counts in sorted(by_language.items(), key=lambda x: -(x[1].tp + x[1].fn)):
            m = compute_metrics(lang_counts)
            support = lang_counts.tp + lang_counts.fn
            print(f"{language:<20} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {support}")

    if by_label_total:
        print("\nGold-label recall")
        print("-" * 60)
        print(f"{'Label':<20} {'Recall':<10} {'Support'}")
        for label, support in by_label_total.most_common():
            recall = by_label_tp[label] / support if support else 0.0
            print(f"{label:<20} {recall:<10.4f} {support}")


def print_comparison_table(results: List[Dict]) -> None:
    print("\nStage Comparison")
    print("=" * 60)
    print(f"{'Stage':<14} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy'}")
    for result in results:
        m = result["metrics"]
        print(
            f"{result['stage']:<14} "
            f"{m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {m['accuracy']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate token-level sensitivity detection accuracy on ai4privacy/pii-masking-300k."
    )
    parser.add_argument("--split", default="train", help="Dataset split to evaluate")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum rows to evaluate")
    parser.add_argument("--language", default=None, help="Optional exact language filter, e.g. English")
    parser.add_argument(
        "--text-limit-chars",
        type=int,
        default=1000,
        help="Trim source_text to this many characters to match calibration behavior",
    )
    parser.add_argument(
        "--cache",
        default=None,
        metavar="PATH",
        help="JSON file to load cached token classifications from and save back to for full-stage runs",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="full",
        help="Detector stage to evaluate",
    )
    parser.add_argument(
        "--compare-stages",
        action="store_true",
        help="Evaluate regex, regex+dict, and full in one run and print a comparison table",
    )
    args = parser.parse_args()

    rows = list(iter_ai4privacy(args.split, args.limit, args.language))
    tokenizer = BPETokenizer()

    stages = STAGES if args.compare_stages else (args.stage,)
    results = []
    for stage in stages:
        cache_file = args.cache if stage == "full" and not args.compare_stages else None
        result = evaluate_stage(
            rows=rows,
            tokenizer=tokenizer,
            stage=stage,
            text_limit_chars=args.text_limit_chars,
            cache_file=cache_file,
        )
        print_summary(result)
        results.append(result)

    if len(results) > 1:
        print_comparison_table(results)


if __name__ == "__main__":
    main()
