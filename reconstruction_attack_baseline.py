"""
Reconstruction Attack — Baseline Detector Comparison
=====================================================
Runs the same LLM reconstruction attack from reconstruction_attack.py
but swaps out the detector (Presidio / spaCy / Flair / Ours) to show
how detector quality affects adversarial reconstruction rates.

Research question:
  A weaker detector encrypts fewer / wrong tokens → leaves more PII in
  plaintext → gives the adversary more context → higher reconstruction rate.
  A stronger detector is harder to reconstruct from.

Expected result table:
  Detector         Tokens encrypted   Exact%   Semantic%   Category%
  spaCy                 low            high      high        high     ← easy to attack
  Flair                 low            high      high        high
  Presidio              medium         medium    medium      medium
  Ours (full)           high           low       low         medium   ← hardest to attack

Install baselines:
    pip install presidio-analyzer flair
    python -m spacy download en_core_web_lg

Usage:
    python reconstruction_attack_baseline.py --model llama3.1:8b --limit 50
    python reconstruction_attack_baseline.py --model llama3.1:8b --limit 50 --detectors spacy ours
"""

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from calibrate import (
    BPETokenizer,
    EMAIL_RE, PHONE_RE, SSN_RE, CREDIT_RE, IP_RE, DATE_RE, TIME_RE,
    Token,
    TokenSensitivityClassifier,
    iter_ai4privacy,
    iter_mtsamples,
)

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

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


TAG_RE = re.compile(r"<ENC:[^>]+>")
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


# ── shared data types ─────────────────────────────────────────────────────────

@dataclass
class SensitiveTokenStub:
    """Minimal duck-type of calibrate.SensitiveToken for baseline detectors."""
    token: Token
    span_type: str
    risk: float
    source: str


@dataclass
class ProtectedValue:
    tag: str
    plaintext: str
    span_type: str
    dataset: str
    source_bucket: str
    token_idx: int
    risk: float


@dataclass
class ProtectedPrompt:
    dataset: str
    source_bucket: str
    detector_name: str
    original_text: str
    protected_text: str
    protected_values: List[ProtectedValue]


# ── baseline detector wrappers ────────────────────────────────────────────────

class OurDetector:
    name = "ours (regex+dict+gliner)"

    def __init__(self, cache_file: Optional[str] = None):
        self.classifier = TokenSensitivityClassifier(cache_file=cache_file, stage="full")

    def classify(self, tokens: List[Token], text: str) -> List[SensitiveTokenStub]:
        results = self.classifier.classify(tokens, text)
        return [SensitiveTokenStub(st.token, st.span_type, st.risk, st.source) for st in results]

    def save_cache(self):
        self.classifier.save_cache()


class PresidioDetector:
    name = "presidio"

    def __init__(self):
        if not _PRESIDIO_AVAILABLE:
            raise RuntimeError("presidio-analyzer not installed: pip install presidio-analyzer")
        self.engine = AnalyzerEngine()

    def classify(self, tokens: List[Token], text: str) -> List[SensitiveTokenStub]:
        results = self.engine.analyze(text=text, language="en")
        sensitive = []
        seen = set()
        for r in results:
            for tok in tokens:
                if tok.idx not in seen and tok.char_start < r.end and tok.char_end > r.start:
                    sensitive.append(SensitiveTokenStub(tok, r.entity_type, r.score, "presidio"))
                    seen.add(tok.idx)
        return sensitive

    def save_cache(self):
        pass


class SpacyDetector:
    name = "spacy (en_core_web_lg)"

    def __init__(self):
        if not _SPACY_AVAILABLE:
            raise RuntimeError("spacy not installed")
        self.nlp = _spacy.load("en_core_web_lg")

    def classify(self, tokens: List[Token], text: str) -> List[SensitiveTokenStub]:
        doc = self.nlp(text)
        sensitive = []
        seen = set()
        for ent in doc.ents:
            for tok in tokens:
                if tok.idx not in seen and tok.char_start < ent.end_char and tok.char_end > ent.start_char:
                    sensitive.append(SensitiveTokenStub(tok, ent.label_, 0.75, "spacy"))
                    seen.add(tok.idx)
        return sensitive

    def save_cache(self):
        pass


class FlairDetector:
    name = "flair (ner-english-large)"

    def __init__(self):
        if not _FLAIR_AVAILABLE:
            raise RuntimeError("flair not installed: pip install flair")
        self.tagger = SequenceTagger.load("flair/ner-english-large")

    def classify(self, tokens: List[Token], text: str) -> List[SensitiveTokenStub]:
        sentence = FlairSentence(text)
        self.tagger.predict(sentence)
        sensitive = []
        seen = set()
        for entity in sentence.get_spans("ner"):
            for tok in tokens:
                if tok.idx not in seen and tok.char_start < entity.end_position and tok.char_end > entity.start_position:
                    sensitive.append(SensitiveTokenStub(tok, entity.tag, entity.score, "flair"))
                    seen.add(tok.idx)
        return sensitive

    def save_cache(self):
        pass


# ── protection logic (same as reconstruction_attack.py) ──────────────────────

def dedupe(sensitive_tokens: List[SensitiveTokenStub]) -> List[SensitiveTokenStub]:
    by_idx = {}
    for st in sensitive_tokens:
        if st.token.idx not in by_idx or st.risk > by_idx[st.token.idx].risk:
            by_idx[st.token.idx] = st
    return [by_idx[i] for i in sorted(by_idx)]


def select_tokens_to_encrypt(sensitive_tokens: List[SensitiveTokenStub], alpha: float) -> List[SensitiveTokenStub]:
    if not sensitive_tokens:
        return []
    alpha = max(0.0, min(1.0, alpha))
    k = max(1, math.ceil(alpha * len(sensitive_tokens)))
    ranked = sorted(sensitive_tokens, key=lambda st: (-st.risk, st.token.idx))
    return sorted(ranked[:k], key=lambda st: st.token.idx)


def make_tag(span_type: str, token_idx: int, plaintext: str) -> str:
    digest = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()[:8]
    return f"<ENC:{span_type}:{token_idx}:{digest}>"


def protect_prompt(dataset: str, source_bucket: str, text: str,
                   tokenizer: BPETokenizer, detector, alpha: float,
                   detector_name: str) -> Optional[ProtectedPrompt]:
    tokens = tokenizer.tokenize(text)
    sensitive = dedupe(detector.classify(tokens, text))
    chosen = {st.token.idx: st for st in select_tokens_to_encrypt(sensitive, alpha)}
    if not chosen:
        return None

    parts: List[str] = []
    protected_values: List[ProtectedValue] = []
    for tok in tokens:
        if tok.idx in chosen:
            st = chosen[tok.idx]
            tag = make_tag(st.span_type, tok.idx, tok.text)
            parts.append(tag)
            protected_values.append(ProtectedValue(
                tag=tag, plaintext=tok.text, span_type=st.span_type,
                dataset=dataset, source_bucket=source_bucket,
                token_idx=tok.idx, risk=st.risk,
            ))
        else:
            parts.append(tok.text)

    return ProtectedPrompt(
        dataset=dataset, source_bucket=source_bucket, detector_name=detector_name,
        original_text=text, protected_text="".join(parts), protected_values=protected_values,
    )


# ── adversary (same Ollama call) ──────────────────────────────────────────────

def build_messages(protected_text: str) -> List[Dict]:
    system = (
        "You are analyzing a partially redacted document. "
        "Tokens marked <ENC:TYPE:idx:hash> have been encrypted. "
        "For each encrypted tag, guess the original plaintext value. "
        "YOU MUST respond with ONLY a valid JSON object and absolutely nothing else — "
        "no explanation, no markdown, no code fences. "
        'Example: {"<ENC:NAME:3:ab12cd34>": "John Smith", "<ENC:EMAIL:7:ef56gh78>": "john@example.com"}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": protected_text}]


def extract_json_dict(raw: str) -> Dict[str, str]:
    raw = raw.strip()
    if not raw:
        raise ValueError("empty model response")

    # strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # try direct parse first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass

    # find outermost {...} with re.DOTALL to handle multiline responses
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in response: {raw[:200]!r}")
    parsed = json.loads(match.group())
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    return {str(k): str(v) for k, v in parsed.items()}


def ollama_chat(model: str, messages: List[Dict], timeout_sec: int) -> str:
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0}}
    req = Request(OLLAMA_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout_sec) as resp:
        return str(json.loads(resp.read())["message"]["content"])


def query_adversary(protected: ProtectedPrompt, model: str, timeout_sec: int, retries: int) -> Dict[str, str]:
    messages = build_messages(protected.protected_text)
    last_error = None
    for attempt in range(retries + 1):
        try:
            return extract_json_dict(ollama_chat(model, messages, timeout_sec))
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.0)
    raise RuntimeError(f"Ollama failed after {retries + 1} attempts: {last_error}")


# ── scoring (same as reconstruction_attack.py) ────────────────────────────────

def normalize(v: str) -> str:
    return v.strip().lower()


def classify_guess_category(value: str) -> str:
    text = value.strip()
    if not text: return "EMPTY"
    if EMAIL_RE.search(text): return "EMAIL"
    if PHONE_RE.search(text): return "PHONE"
    if SSN_RE.search(text): return "SSN"
    if CREDIT_RE.search(text): return "CREDIT_CARD"
    if IP_RE.search(text): return "IP"
    if DATE_RE.search(text): return "DATE"
    if TIME_RE.search(text): return "TIME"
    if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text): return "NAME"
    if re.fullmatch(r"[A-Za-z0-9._-]{4,}", text): return "ALNUM_ID"
    if re.fullmatch(r"\d[\d\- ]{3,}", text): return "NUMBER"
    return "TEXT"


def semantic_match(guess: str, truth: str, category: str) -> bool:
    ng, nt = normalize(guess), normalize(truth)
    if not ng or not nt: return False
    if ng == nt: return True
    if category in {"NAME", "ORG", "LOCATION"} and ng[:3] == nt[:3]: return True
    return SequenceMatcher(None, ng, nt).ratio() >= 0.8


def score_guess(guess: str, target: ProtectedValue) -> Dict:
    exact = normalize(guess) == normalize(target.plaintext)
    category = classify_guess_category(target.plaintext)
    return {
        "exact": exact,
        "semantic": semantic_match(guess, target.plaintext, category),
        "category": classify_guess_category(guess) == category,
    }


def aggregate(records: List[Dict], group_key: str) -> Dict:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        g = rec[group_key]
        grouped[g]["n"] += 1
        grouped[g]["exact"] += int(rec["exact"])
        grouped[g]["semantic"] += int(rec["semantic"])
        grouped[g]["category"] += int(rec["category"])
    return {
        g: {
            "n_tags": c["n"],
            "exact_rate": c["exact"] / c["n"],
            "semantic_rate": c["semantic"] / c["n"],
            "category_rate": c["category"] / c["n"],
        }
        for g, c in grouped.items()
    }


# ── per-detector experiment ───────────────────────────────────────────────────

def run_detector_experiment(detector, sources: List[Tuple], tokenizer: BPETokenizer,
                             alpha: float, model: str, timeout: int, retries: int) -> Dict:
    prompts: List[ProtectedPrompt] = []
    for dataset, bucket, text in sources:
        p = protect_prompt(dataset, bucket, text, tokenizer, detector, alpha, detector.name)
        if p and p.protected_values:
            prompts.append(p)

    detector.save_cache()

    n_total_tokens = sum(len(tokenizer.tokenize(p.original_text)) for p in prompts)
    n_encrypted = sum(len(p.protected_values) for p in prompts)

    records: List[Dict] = []
    prompt_iter = tqdm(prompts, desc=f"{detector.name} — attack", unit="prompt") if _TQDM_AVAILABLE else prompts
    for prompt in prompt_iter:
        try:
            guesses = query_adversary(prompt, model, timeout, retries)
        except RuntimeError as e:
            print(f"  Skipping prompt: {e}")
            continue
        for target in prompt.protected_values:
            guess = guesses.get(target.tag, "")
            score = score_guess(guess, target)
            records.append({
                "detector": detector.name,
                "dataset": target.dataset,
                "span_type": target.span_type,
                "truth": target.plaintext,
                "guess": guess,
                **score,
            })

    overall = aggregate(records, "detector").get(detector.name, {})
    return {
        "detector": detector.name,
        "n_prompts": len(prompts),
        "n_total_tokens": n_total_tokens,
        "n_encrypted": n_encrypted,
        "encrypt_rate": n_encrypted / n_total_tokens if n_total_tokens else 0.0,
        "overall": overall,
        "by_span_type": aggregate(records, "span_type"),
        "records": records,
    }


# ── output ────────────────────────────────────────────────────────────────────

def print_comparison_table(results: List[Dict]) -> None:
    print("\n" + "=" * 80)
    print("Reconstruction Attack — Detector Comparison")
    print("=" * 80)
    print(f"{'Detector':<32} {'Encrypt%':<10} {'Exact%':<10} {'Semantic%':<10} {'Category%'}")
    print("-" * 80)
    for r in results:
        o = r["overall"]
        print(
            f"{r['detector']:<32} "
            f"{r['encrypt_rate']:<10.2%} "
            f"{o.get('exact_rate', 0):<10.2%} "
            f"{o.get('semantic_rate', 0):<10.2%} "
            f"{o.get('category_rate', 0):.2%}"
        )
    print("=" * 80)
    print("\nNote: lower Exact%/Semantic% = harder to reconstruct = better privacy protection")


# ── main ──────────────────────────────────────────────────────────────────────

ALL_DETECTORS = ["presidio", "spacy", "flair", "ours"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare detector quality vs adversarial reconstruction rate."
    )
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--mtsamples-limit", type=int, default=50)
    parser.add_argument("--ai4privacy-limit", type=int, default=50)
    parser.add_argument("--ai4privacy-split", default="train")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default="reconstruction_baseline_results.json")
    parser.add_argument("--detectors", nargs="+", default=ALL_DETECTORS, choices=ALL_DETECTORS)
    args = parser.parse_args()

    # build source list once, shared across all detectors
    sources = []
    for bucket, text in iter_mtsamples(args.mtsamples_limit):
        sources.append(("mtsamples", bucket, text))
    if args.ai4privacy_limit:
        for bucket, text in iter_ai4privacy(args.ai4privacy_limit, args.ai4privacy_split):
            sources.append(("ai4privacy", bucket, text))

    tokenizer = BPETokenizer()

    detectors = []
    for name in args.detectors:
        if name == "presidio":
            if not _PRESIDIO_AVAILABLE:
                print("Skipping presidio — pip install presidio-analyzer")
                continue
            detectors.append(PresidioDetector())
        elif name == "spacy":
            if not _SPACY_AVAILABLE:
                print("Skipping spacy — pip install spacy && python -m spacy download en_core_web_lg")
                continue
            try:
                detectors.append(SpacyDetector())
            except OSError:
                print("Skipping spacy — en_core_web_lg not found")
        elif name == "flair":
            if not _FLAIR_AVAILABLE:
                print("Skipping flair — pip install flair")
                continue
            detectors.append(FlairDetector())
        elif name == "ours":
            detectors.append(OurDetector(cache_file=args.cache))

    if not detectors:
        print("No detectors available.")
        return

    all_results = []
    for detector in detectors:
        print(f"\n{'='*60}")
        print(f"Running: {detector.name}")
        print(f"{'='*60}")
        result = run_detector_experiment(
            detector, sources, tokenizer,
            args.alpha, args.model, args.timeout, args.retries,
        )
        all_results.append(result)

    print_comparison_table(all_results)

    with open(args.output, "w") as f:
        # exclude per-record details to keep file manageable
        summary = [{k: v for k, v in r.items() if k != "records"} for r in all_results]
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
