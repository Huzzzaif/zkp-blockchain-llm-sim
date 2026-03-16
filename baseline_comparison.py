"""
Baseline comparison: Presidio vs spaCy vs Flair vs Our pipeline
===============================================================
Runs the same token-level eval loop from sensitivity_accuracy_experiment.py
against three established baselines on ai4privacy/pii-masking-300k.

Install baselines:
    pip install presidio-analyzer presidio-nlp-engine
    python -m spacy download en_core_web_lg
    pip install flair

Usage:
    python baseline_comparison.py --limit 500
    python baseline_comparison.py --limit 500 --systems spacy flair ours
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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

# ── optional baselines ────────────────────────────────────────────────────────

try:
    from presidio_analyzer import AnalyzerEngine
    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False

try:
    import spacy as _spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    from flair.models import SequenceTagger
    from flair.data import Sentence as FlairSentence
    _FLAIR_AVAILABLE = True
except ImportError:
    _FLAIR_AVAILABLE = False


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class EvalCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0


# ── shared eval helpers ───────────────────────────────────────────────────────

def token_overlaps_span(tok: Token, start: int, end: int) -> bool:
    return tok.char_start < end and tok.char_end > start


def gold_token_labels(tokens: List[Token], privacy_mask: List[Dict]) -> Tuple[set, Dict]:
    gold_positive: set = set()
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
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def score_tokens(tokens: List[Token], predicted_positive: set, gold_positive: set,
                 gold_label_by_idx: Dict, total: EvalCounts,
                 by_label_total: Counter, by_label_tp: Counter) -> None:
    for tok in tokens:
        pred = tok.idx in predicted_positive
        gold = tok.idx in gold_positive
        if pred and gold:
            total.tp += 1
            by_label_tp[gold_label_by_idx[tok.idx]] += 1
        elif pred and not gold:
            total.fp += 1
        elif not pred and gold:
            total.fn += 1
        else:
            total.tn += 1
        if gold:
            by_label_total[gold_label_by_idx[tok.idx]] += 1


# ── dataset loader ────────────────────────────────────────────────────────────

def iter_ai4privacy(split: str, limit: Optional[int], language: Optional[str]) -> List[Dict]:
    if not _DATASETS_AVAILABLE:
        raise RuntimeError("datasets not installed")
    ds = load_dataset("ai4privacy/pii-masking-300k", split=split)
    if language:
        ds = ds.filter(lambda r: str(r.get("language", "")).lower() == language.lower())
    else:
        ds = ds.filter(lambda r: str(r.get("language", "")).lower() in ("english", "spanish"))
    ds = ds.shuffle(seed=42)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return list(ds)


# ── baseline detectors ────────────────────────────────────────────────────────

class PresidioDetector:
    name = "presidio"

    def __init__(self):
        if not _PRESIDIO_AVAILABLE:
            raise RuntimeError("presidio-analyzer not installed: pip install presidio-analyzer")
        self.engine = AnalyzerEngine()

    def sensitive_token_indices(self, tokens: List[Token], text: str) -> set:
        results = self.engine.analyze(text=text, language="en")
        positive = set()
        for r in results:
            for tok in tokens:
                if token_overlaps_span(tok, r.start, r.end):
                    positive.add(tok.idx)
        return positive


class SpacyDetector:
    name = "spacy (en_core_web_lg)"

    def __init__(self):
        if not _SPACY_AVAILABLE:
            raise RuntimeError("spacy not installed: pip install spacy && python -m spacy download en_core_web_lg")
        self.nlp = _spacy.load("en_core_web_lg")

    def sensitive_token_indices(self, tokens: List[Token], text: str) -> set:
        doc = self.nlp(text)
        positive = set()
        for ent in doc.ents:
            for tok in tokens:
                if token_overlaps_span(tok, ent.start_char, ent.end_char):
                    positive.add(tok.idx)
        return positive


class FlairDetector:
    name = "flair (ner-english-large)"

    def __init__(self):
        if not _FLAIR_AVAILABLE:
            raise RuntimeError("flair not installed: pip install flair")
        self.tagger = SequenceTagger.load("flair/ner-english-large")

    def sensitive_token_indices(self, tokens: List[Token], text: str) -> set:
        sentence = FlairSentence(text)
        self.tagger.predict(sentence)
        positive = set()
        for entity in sentence.get_spans("ner"):
            for tok in tokens:
                if token_overlaps_span(tok, entity.start_position, entity.end_position):
                    positive.add(tok.idx)
        return positive


class OurDetector:
    name = "ours (regex+dict+gliner)"

    def __init__(self, cache_file: Optional[str] = None):
        self.classifier = TokenSensitivityClassifier(cache_file=cache_file, stage="full")

    def sensitive_token_indices(self, tokens: List[Token], text: str, language: str = "") -> set:
        results = self.classifier.classify(tokens, text, language=language)
        return {st.token.idx for st in results}

    def save_cache(self):
        self.classifier.save_cache()


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_detector(detector, rows: List[Dict], tokenizer: BPETokenizer,
                      text_limit_chars: int) -> Dict:
    total = EvalCounts()
    by_label_total: Counter = Counter()
    by_label_tp: Counter = Counter()
    n_examples = 0

    row_iter = tqdm(rows, desc=detector.name, unit="row") if _TQDM_AVAILABLE else rows

    for row in row_iter:
        text = str(row.get("source_text", ""))
        if not text.strip():
            continue
        text = text[:text_limit_chars]
        privacy_mask = [s for s in row.get("privacy_mask", []) if int(s["start"]) < len(text)]

        tokens = tokenizer.tokenize(text)
        language = str(row.get("language", ""))

        if isinstance(detector, OurDetector):
            predicted_positive = detector.sensitive_token_indices(tokens, text, language)
        else:
            predicted_positive = detector.sensitive_token_indices(tokens, text)

        gold_positive, gold_label_by_idx = gold_token_labels(tokens, privacy_mask)
        score_tokens(tokens, predicted_positive, gold_positive, gold_label_by_idx,
                     total, by_label_total, by_label_tp)
        n_examples += 1

    if isinstance(detector, OurDetector):
        detector.save_cache()

    return {
        "system": detector.name,
        "counts": total,
        "metrics": compute_metrics(total),
        "by_label_total": by_label_total,
        "by_label_tp": by_label_tp,
        "n_examples": n_examples,
    }


# ── output ────────────────────────────────────────────────────────────────────

def print_comparison_table(results: List[Dict]) -> None:
    print("\n" + "=" * 72)
    print("Baseline Comparison — Token-Level PII Detection")
    print("=" * 72)
    print(f"{'System':<32} {'Precision':<11} {'Recall':<11} {'F1':<11} {'Accuracy'}")
    print("-" * 72)
    for r in results:
        m = r["metrics"]
        print(f"{r['system']:<32} {m['precision']:<11.4f} {m['recall']:<11.4f} "
              f"{m['f1']:<11.4f} {m['accuracy']:.4f}")
    print("=" * 72)


def print_label_recall(result: Dict) -> None:
    by_label_total = result["by_label_total"]
    by_label_tp = result["by_label_tp"]
    if not by_label_total:
        return
    print(f"\n  Gold-label recall [{result['system']}]")
    print(f"  {'Label':<22} {'Recall':<10} {'Support'}")
    for label, support in by_label_total.most_common(15):
        recall = by_label_tp[label] / support if support else 0.0
        print(f"  {label:<22} {recall:<10.4f} {support}")


# ── main ──────────────────────────────────────────────────────────────────────

ALL_SYSTEMS = ["presidio", "spacy", "flair", "ours"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Presidio / spaCy / Flair / Our pipeline on ai4privacy/pii-masking-300k."
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=500, help="Rows to evaluate (default 500)")
    parser.add_argument("--language", default=None, help="Optional language filter e.g. English")
    parser.add_argument("--text-limit-chars", type=int, default=1000)
    parser.add_argument("--cache", default=None, metavar="PATH", help="Token cache for our detector")
    parser.add_argument(
        "--systems", nargs="+", default=ALL_SYSTEMS,
        choices=ALL_SYSTEMS,
        help="Which systems to run (default: all)"
    )
    args = parser.parse_args()

    print(f"Loading {args.limit} rows from ai4privacy ({args.split})...")
    rows = iter_ai4privacy(args.split, args.limit, args.language)
    tokenizer = BPETokenizer()

    detectors = []
    for name in args.systems:
        if name == "presidio":
            if not _PRESIDIO_AVAILABLE:
                print("Skipping presidio — not installed (pip install presidio-analyzer)")
                continue
            detectors.append(PresidioDetector())
        elif name == "spacy":
            if not _SPACY_AVAILABLE:
                print("Skipping spacy — not installed (pip install spacy && python -m spacy download en_core_web_lg)")
                continue
            try:
                detectors.append(SpacyDetector())
            except OSError:
                print("Skipping spacy — en_core_web_lg model not found (python -m spacy download en_core_web_lg)")
        elif name == "flair":
            if not _FLAIR_AVAILABLE:
                print("Skipping flair — not installed (pip install flair)")
                continue
            detectors.append(FlairDetector())
        elif name == "ours":
            detectors.append(OurDetector(cache_file=args.cache))

    if not detectors:
        print("No detectors available. Install at least one baseline.")
        return

    results = []
    for detector in detectors:
        result = evaluate_detector(detector, rows, tokenizer, args.text_limit_chars)
        results.append(result)

    print_comparison_table(results)
    for r in results:
        print_label_recall(r)


if __name__ == "__main__":
    main()
