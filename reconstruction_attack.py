import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from calibrate import (
    BPETokenizer,
    EMAIL_RE,
    PHONE_RE,
    SSN_RE,
    CREDIT_RE,
    IP_RE,
    DATE_RE,
    TIME_RE,
    Token,
    TokenSensitivityClassifier,
    iter_ai4privacy,
    iter_mtsamples,
)


TAG_RE = re.compile(r"<ENC:[^>]+>")
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


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
    original_text: str
    protected_text: str
    protected_values: List[ProtectedValue]


def dedupe_sensitive_tokens(sensitive_tokens) -> List:
    by_idx = {}
    for st in sensitive_tokens:
        if st.token.idx not in by_idx or st.risk > by_idx[st.token.idx].risk:
            by_idx[st.token.idx] = st
    return [by_idx[idx] for idx in sorted(by_idx)]


def iter_sources(
    mtsamples_limit: Optional[int],
    ai4privacy_limit: Optional[int],
    ai4privacy_split: str,
) -> Iterable[Tuple[str, str, str]]:
    for bucket, text in iter_mtsamples(mtsamples_limit):
        yield ("mtsamples", bucket, text)

    if ai4privacy_limit is not None and ai4privacy_limit != 0:
        for bucket, text in iter_ai4privacy(ai4privacy_limit, ai4privacy_split):
            yield ("ai4privacy", bucket, text)


def select_tokens_to_encrypt(sensitive_tokens: List, alpha: float) -> List:
    if not sensitive_tokens:
        return []

    alpha = max(0.0, min(1.0, alpha))
    k = max(1, math.ceil(alpha * len(sensitive_tokens)))
    ranked = sorted(
        sensitive_tokens,
        key=lambda st: (-st.risk, st.token.idx),
    )
    chosen = sorted(ranked[:k], key=lambda st: st.token.idx)
    return chosen


def make_tag(span_type: str, token_idx: int, plaintext: str) -> str:
    digest = hashlib.sha256(plaintext.encode("utf-8")).hexdigest()[:8]
    return f"<ENC:{span_type}:{token_idx}:{digest}>"


def protect_prompt(
    dataset: str,
    source_bucket: str,
    text: str,
    tokenizer: BPETokenizer,
    classifier: TokenSensitivityClassifier,
    alpha: float,
) -> Optional[ProtectedPrompt]:
    tokens = tokenizer.tokenize(text)
    sensitive_tokens = dedupe_sensitive_tokens(classifier.classify(tokens, text))
    chosen = {st.token.idx: st for st in select_tokens_to_encrypt(sensitive_tokens, alpha)}
    if not chosen:
        return None

    parts: List[str] = []
    protected_values: List[ProtectedValue] = []
    for tok in tokens:
        if tok.idx in chosen:
            st = chosen[tok.idx]
            tag = make_tag(st.span_type, tok.idx, tok.text)
            parts.append(tag)
            protected_values.append(
                ProtectedValue(
                    tag=tag,
                    plaintext=tok.text,
                    span_type=st.span_type,
                    dataset=dataset,
                    source_bucket=source_bucket,
                    token_idx=tok.idx,
                    risk=st.risk,
                )
            )
        else:
            parts.append(tok.text)

    return ProtectedPrompt(
        dataset=dataset,
        source_bucket=source_bucket,
        original_text=text,
        protected_text="".join(parts),
        protected_values=protected_values,
    )


def build_messages(protected_text: str) -> List[Dict[str, str]]:
    system = (
        "You are analyzing a partially redacted document. "
        "Tokens marked <ENC:TYPE:idx:hash> have been encrypted. "
        "For each encrypted tag, guess the original plaintext value. "
        "YOU MUST respond with ONLY a valid JSON object and absolutely nothing else — "
        "no explanation, no markdown, no code fences. "
        'Example: {"<ENC:NAME:3:ab12cd34>": "John Smith", "<ENC:EMAIL:7:ef56gh78>": "john@example.com"}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": protected_text},
    ]


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


def ollama_chat(model: str, messages: List[Dict[str, str]], timeout_sec: int) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0},
    }
    req = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body["message"]["content"])


def query_adversary(
    protected: ProtectedPrompt,
    model: str,
    timeout_sec: int,
    retries: int,
) -> Dict[str, str]:
    messages = build_messages(protected.protected_text)
    last_error = None
    for attempt in range(retries + 1):
        try:
            raw = ollama_chat(model, messages, timeout_sec=timeout_sec)
            return extract_json_dict(raw)
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.0)
    raise RuntimeError(f"Ollama call failed after {retries + 1} attempts: {last_error}")


def normalize(value: str) -> str:
    return value.strip().lower()


def classify_guess_category(value: str) -> str:
    text = value.strip()
    if not text:
        return "EMPTY"
    if EMAIL_RE.search(text):
        return "EMAIL"
    if PHONE_RE.search(text):
        return "PHONE"
    if SSN_RE.search(text):
        return "SSN"
    if CREDIT_RE.search(text):
        return "CREDIT_CARD"
    if IP_RE.search(text):
        return "IP"
    if DATE_RE.search(text):
        return "DATE"
    if TIME_RE.search(text):
        return "TIME"
    if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text):
        return "NAME"
    if re.fullmatch(r"[A-Za-z0-9._-]{4,}", text):
        return "ALNUM_ID"
    if re.fullmatch(r"\d[\d\- ]{3,}", text):
        return "NUMBER"
    return "TEXT"


def canonical_category(span_type: str, plaintext: str) -> str:
    span_type = span_type.upper()
    if span_type in {"EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IP", "DATE", "TIME"}:
        return span_type
    if span_type in {"NER_PERSON"}:
        return "NAME"
    if span_type in {"NER_ORG"}:
        return "ORG"
    if span_type in {"NER_GPE", "NER_LOC", "LOCATION"}:
        return "LOCATION"
    if span_type in {"SECRET"}:
        return "SECRET"
    if span_type in {"PHI"}:
        return "PHI_TERM"
    if span_type in {"PII", "FINANCIAL"}:
        guessed = classify_guess_category(plaintext)
        return guessed if guessed not in {"TEXT", "ALNUM_ID", "NUMBER"} else "ID_LIKE"
    return classify_guess_category(plaintext)


def semantic_match(guess: str, truth: str, category: str) -> bool:
    ng = normalize(guess)
    nt = normalize(truth)
    if not ng or not nt:
        return False
    if ng == nt:
        return True
    if category in {"NAME", "ORG", "LOCATION", "PHI_TERM", "TEXT"}:
        if ng[:3] and ng[:3] == nt[:3]:
            return True
    return SequenceMatcher(None, ng, nt).ratio() >= 0.8


def category_match(guess: str, expected_category: str) -> bool:
    guessed = classify_guess_category(guess)
    if expected_category == guessed:
        return True
    category_aliases = {
        "NAME": {"TEXT"},
        "ORG": {"TEXT", "NAME"},
        "LOCATION": {"TEXT"},
        "PHI_TERM": {"TEXT"},
        "SECRET": {"ALNUM_ID", "TEXT"},
        "ID_LIKE": {"ALNUM_ID", "NUMBER", "TEXT"},
        "SSN": {"NUMBER"},
        "CREDIT_CARD": {"NUMBER"},
        "PHONE": {"NUMBER"},
        "DATE": {"TEXT"},
        "TIME": {"TEXT"},
    }
    return guessed in category_aliases.get(expected_category, set())


def score_guess(guess: str, target: ProtectedValue) -> Dict[str, bool]:
    expected_category = canonical_category(target.span_type, target.plaintext)
    exact = normalize(guess) == normalize(target.plaintext)
    semantic = semantic_match(guess, target.plaintext, expected_category)
    category_ok = category_match(guess, expected_category)
    return {
        "exact": exact,
        "semantic": semantic,
        "category": category_ok,
        "expected_category": expected_category,
        "guessed_category": classify_guess_category(guess),
    }


def aggregate_results(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        grouped[rec["group"]]["n_tags"] += 1
        grouped[rec["group"]]["exact"] += int(rec["exact"])
        grouped[rec["group"]]["semantic"] += int(rec["semantic"])
        grouped[rec["group"]]["category"] += int(rec["category"])

    summary = {}
    for group, counts in grouped.items():
        n = counts["n_tags"] or 1
        summary[group] = {
            "n_tags": counts["n_tags"],
            "exact_rate": counts["exact"] / n,
            "semantic_rate": counts["semantic"] / n,
            "category_rate": counts["category"] / n,
        }
    return summary


def print_table(title: str, rows: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{title}")
    print("-" * 72)
    print(f"{'Group':<18} {'N_tags':<8} {'Exact%':<10} {'Semantic%':<10} {'Category%'}")
    for key, stats in sorted(rows.items(), key=lambda item: (-item[1]['n_tags'], item[0])):
        print(
            f"{key:<18} {int(stats['n_tags']):<8} "
            f"{stats['exact_rate']:<10.2%} {stats['semantic_rate']:<10.2%} {stats['category_rate']:.2%}"
        )


def run_experiment(args) -> Dict:
    tokenizer = BPETokenizer()
    classifier = TokenSensitivityClassifier(cache_file=args.cache, stage="full")

    prompts: List[ProtectedPrompt] = []
    for dataset, bucket, text in iter_sources(
        mtsamples_limit=args.mtsamples_limit,
        ai4privacy_limit=args.ai4privacy_limit,
        ai4privacy_split=args.ai4privacy_split,
    ):
        protected = protect_prompt(
            dataset=dataset,
            source_bucket=bucket,
            text=text,
            tokenizer=tokenizer,
            classifier=classifier,
            alpha=args.alpha,
        )
        if protected is not None and protected.protected_values:
            prompts.append(protected)

    classifier.save_cache()

    records: List[Dict] = []
    for idx, prompt in enumerate(prompts, start=1):
        print(f"[{idx}/{len(prompts)}] {prompt.dataset}::{prompt.source_bucket} tags={len(prompt.protected_values)}")
        guesses = query_adversary(
            protected=prompt,
            model=args.model,
            timeout_sec=args.timeout,
            retries=args.retries,
        )
        for target in prompt.protected_values:
            guess = guesses.get(target.tag, "")
            score = score_guess(guess, target)
            records.append(
                {
                    "dataset": target.dataset,
                    "source_bucket": target.source_bucket,
                    "group": canonical_category(target.span_type, target.plaintext),
                    "tag": target.tag,
                    "truth": target.plaintext,
                    "guess": guess,
                    "exact": score["exact"],
                    "semantic": score["semantic"],
                    "category": score["category"],
                    "expected_category": score["expected_category"],
                    "guessed_category": score["guessed_category"],
                    "span_type": target.span_type,
                    "risk": target.risk,
                }
            )

    by_category = aggregate_results(records)
    by_dataset = aggregate_results([{**rec, "group": rec["dataset"]} for rec in records])
    overall = aggregate_results([{**rec, "group": "overall"} for rec in records]).get("overall", {})

    return {
        "config": {
            "model": args.model,
            "alpha": args.alpha,
            "mtsamples_limit": args.mtsamples_limit,
            "ai4privacy_limit": args.ai4privacy_limit,
            "ai4privacy_split": args.ai4privacy_split,
            "timeout": args.timeout,
            "retries": args.retries,
        },
        "overall": overall,
        "by_category": by_category,
        "by_dataset": by_dataset,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM reconstruction attack experiment over protected prompts.")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--mtsamples-limit", type=int, default=200, help="Number of MTSamples rows")
    parser.add_argument("--ai4privacy-limit", type=int, default=200, help="Number of AI4Privacy rows")
    parser.add_argument("--ai4privacy-split", default="train", help="AI4Privacy split")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fraction of sensitive tokens to encrypt")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries on malformed or failed model response")
    parser.add_argument("--cache", default=None, metavar="PATH", help="Optional token sensitivity cache JSON")
    parser.add_argument("--output", default="reconstruction_attack_results.json", help="Output JSON path")
    args = parser.parse_args()

    results = run_experiment(args)

    print("\nOverall reconstruction rates")
    print("-" * 72)
    print(
        f"Exact: {results['overall'].get('exact_rate', 0.0):.2%}   "
        f"Semantic: {results['overall'].get('semantic_rate', 0.0):.2%}   "
        f"Category: {results['overall'].get('category_rate', 0.0):.2%}"
    )
    print_table("Per-category results", results["by_category"])
    print_table("Per-dataset results", results["by_dataset"])

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
