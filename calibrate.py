import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

try:
    from gliner import GLiNER
    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False

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


@dataclass(frozen=True, eq=True)
class Token:
    idx: int
    text: str
    char_start: int
    char_end: int


@dataclass
class SensitiveToken:
    token: Token
    span_type: str
    risk: float
    source: str


class BPETokenizer:
    def __init__(self):
        self.enc = None
        if _TIKTOKEN_AVAILABLE:
            try:
                self.enc = tiktoken.get_encoding("cl100k_base")
                print("Loaded tiktoken cl100k_base")
            except Exception:
                self.enc = None
        if self.enc is None:
            print("tiktoken unavailable; using fallback tokenizer")

    def tokenize(self, text: str) -> list[Token]:
        if self.enc is not None:
            return self._tiktoken_tokenize(text)
        return self._fallback_tokenize(text)

    def _tiktoken_tokenize(self, text: str) -> list[Token]:
        token_ids = self.enc.encode(text)
        tokens = []
        char_pos = 0
        for idx, tid in enumerate(token_ids):
            token_str = self.enc.decode([tid])
            char_start = text.find(token_str, char_pos)
            if char_start == -1:
                char_start = char_pos
            char_end = char_start + len(token_str)
            tokens.append(Token(idx, token_str, char_start, char_end))
            char_pos = char_end
        return tokens

    def _fallback_tokenize(self, text: str) -> list[Token]:
        pattern = re.compile(r"[A-Za-z0-9]+|[^\s\w]|\s+")
        return [
            Token(idx, match.group(), match.start(), match.end())
            for idx, match in enumerate(pattern.finditer(text))
        ]


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
IPV6_RE = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"
    r"|\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}\b"
    r"|\b::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b"
)
_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}"                           # ISO: 2024-03-05
    r"|\d{1,2}/\d{1,2}/\d{2,4}"                    # US:  03/05/2024
    r"|\d{1,2}-\d{1,2}-\d{4}"                      # EU:  05-03-2024
    r"|\d{1,2}\.\d{1,2}\.\d{2,4}"                  # dot: 05.03.2024
    rf"|{_MONTHS}\.?\s+\d{{1,2}},?\s+\d{{4}}"      # March 5, 2024
    rf"|\d{{1,2}}\s+{_MONTHS}\.?\s+\d{{4}}"        # 5 March 2024
    rf"|{_MONTHS}\.?\s+\d{{4}}"                     # March 2024 (no day)
    r")\b",
    re.I,
)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm])?\b")

RISK_TERMS: Dict[str, Tuple[str, float]] = {
    "password": ("SECRET", 1.00),
    "passwd": ("SECRET", 1.00),
    "api_key": ("SECRET", 1.00),
    "api key": ("SECRET", 1.00),
    "secret": ("SECRET", 0.90),
    "token": ("SECRET", 0.85),
    "credit card": ("FINANCIAL", 0.95),
    "ssn": ("PII", 1.00),
}

GLINER_LABELS = (
    "person",
    "organization",
    "location",
    "username",
    "password",
    "email",
    "phone number",
    "passport number",
    "driver license",
    "id card number",
    "street address",
    "postal code",
    "city",
    "state",
    "date of birth",
)

CONTEXT_PATTERNS = [
    (re.compile(r"\b(my|his|her|their)\s+\w+", re.I), 0.70),
    (re.compile(r"\breset\s+\w+", re.I), 0.75),
    (re.compile(r"\bneed\s+\w+", re.I), 0.65),
]

SPANISH_CONTEXT_PATTERNS = [
    (re.compile(r"\b(mi|su)\s+\w+", re.I), 0.70),
    (re.compile(r"\brestablecer\s+\w+", re.I), 0.75),
    (re.compile(r"\bnecesito\s+\w+", re.I), 0.65),
]

# Context triggers: match a keyword followed by the actual PII value (captured in group 1).
# Used for labels with low standalone recall: PASS, USERNAME, DRIVERLICENSE, SEX, STATE, TITLE.
CONTEXT_TRIGGERS: list[Tuple[re.Pattern, str, float]] = [
    (re.compile(r"\b(?:password|passwd)\s*[=:\s]\s*(\S+)", re.I),                       "SECRET",   0.95),
    (re.compile(r"\busername\s*[=:\s]\s*(\S+)", re.I),                                  "SECRET",   0.90),
    (re.compile(r"\bdriver['\s]*s?\s*licen[sc]e\s*(?:number|no\.?|#|:)?\s*([A-Z0-9][A-Z0-9\-]{3,})", re.I), "PII", 0.90),
    (re.compile(r"\b(?:gender|sex)\s*[:\-]?\s*(male|female|non[\-\s]binary|[MF])\b", re.I), "PII", 0.75),
    (re.compile(r"\b(?:state|province)\s*[:\-]?\s*([A-Z]{2})\b"),                       "LOCATION", 0.70),
    (re.compile(r"\b(?:title|salutation|prefix)\s*[:\-]?\s*(Mr|Mrs|Ms|Dr|Prof|Miss)\b", re.I), "PII", 0.65),
]

SPANISH_CONTEXT_TRIGGERS: list[Tuple[re.Pattern, str, float]] = [
    (re.compile(r"\b(?:contrasena|contraseña)\s*[=:\s]\s*(\S+)", re.I), "SECRET", 0.95),
    (re.compile(r"\busuario\s*[=:\s]\s*(\S+)", re.I), "SECRET", 0.90),
    (re.compile(r"\b(?:licencia\s+de\s+conducir|carnet)\s*(?:numero|n[uú]mero|no\.?|#|:)?\s*([A-Z0-9][A-Z0-9\-]{3,})", re.I), "PII", 0.90),
    (re.compile(r"\b(?:sexo|genero|género)\s*[:\-]?\s*(masculino|femenino|[MF])\b", re.I), "PII", 0.75),
    (re.compile(r"\b(?:estado|provincia)\s*[:\-]?\s*([A-Z]{2})\b"), "LOCATION", 0.70),
    (re.compile(r"\b(?:titulo|título|tratamiento)\s*[:\-]?\s*(Sr|Sra|Srta|Dr|Dra|Prof)\b", re.I), "PII", 0.65),
    (re.compile(r"\bciudad\s*[:\-]?\s*([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ]+)*)", re.I), "LOCATION", 0.70),
    (re.compile(r"\b(?:codigo\s+postal|código\s+postal)\s*[:\-]?\s*([A-Z0-9\-]{4,10})", re.I), "PII", 0.80),
    (re.compile(r"\bpasaporte\s*(?:numero|n[uú]mero|no\.?|#|:)?\s*([A-Z0-9][A-Z0-9\-]{5,})", re.I), "PII", 0.90),
]


class TokenSensitivityClassifier:
    def __init__(self, cache_file: str | None = None, stage: str = "full"):
        if stage not in {"regex", "regex+dict", "full"}:
            raise ValueError(f"Unsupported stage: {stage}")
        self.word_cache: Dict[str, Tuple[str, float, str]] = {}
        self.doc_span_cache: Dict[str, list] = {}
        self.stats = {"cache_hits": 0, "cache_misses": 0, "nlp_calls": 0}
        self.gliner_model = None
        self.stage = stage
        self.cache_file = cache_file if stage == "full" else None
        self._init_cache()
        self._load_gliner()

    def _init_cache(self) -> None:
        if self.cache_file and Path(self.cache_file).exists():
            with open(self.cache_file) as f:
                saved = json.load(f)
            loaded = 0
            for word, entry in saved.items():
                if (
                    isinstance(entry, list)
                    and len(entry) == 3
                    and entry[2] == "regex_cache"
                    and word not in self.word_cache
                ):
                    self.word_cache[word] = tuple(entry)
                    loaded += 1
            print(f"Loaded {loaded} entries from cache file: {self.cache_file}")

    def save_cache(self) -> None:
        if not self.cache_file:
            return
        serializable = {word: list(val) for word, val in self.word_cache.items()}
        with open(self.cache_file, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved {len(serializable)} cache entries to: {self.cache_file}")

    def _load_gliner(self) -> None:
        if self.stage != "full":
            return
        if not _GLINER_AVAILABLE:
            print("GLiNER unavailable; model fallback disabled")
            return
        try:
            self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
            print("Loaded GLiNER urchade/gliner_multi_pii-v1")
        except Exception as exc:
            self.gliner_model = None
            print(f"GLiNER load failed; model fallback disabled ({exc})")

    def _detect_language(self, text: str, language_hint: Optional[str] = None) -> str:
        if language_hint:
            language_hint = language_hint.strip().lower()
            if language_hint.startswith("en"):
                return "english"
            if language_hint.startswith("sp") or language_hint.startswith("es"):
                return "spanish"
        lower = text.lower()
        if re.search(r"[áéíóúñ]", lower):
            return "spanish"
        spanish_hits = sum(1 for w in (" el ", " la ", " de ", " que ", " y ", " para ", " con ", " una ", " por ") if w in f" {lower} ")
        english_hits = sum(1 for w in (" the ", " and ", " for ", " with ", " you ", " your ", " from ", " this ") if w in f" {lower} ")
        return "spanish" if spanish_hits > english_hits else "english"

    def _cache_key(self, language: str, word: str) -> str:
        return f"{language}::{word}"

    def _language_context_triggers(self, language: str) -> list[Tuple[re.Pattern, str, float]]:
        return SPANISH_CONTEXT_TRIGGERS if language == "spanish" else CONTEXT_TRIGGERS

    def _regex_char_spans(self, text: str) -> list[Tuple[int, int, str, float]]:
        spans = []
        for regex, span_type, risk in (
            (EMAIL_RE,  "EMAIL",       0.95),
            (PHONE_RE,  "PHONE",       0.90),
            (SSN_RE,    "SSN",         1.00),
            (CREDIT_RE, "CREDIT_CARD", 0.95),
            (IP_RE,     "IP",          0.85),
            (IPV6_RE,   "IP",          0.85),
            (DATE_RE,   "DATE",        0.80),
            (TIME_RE,   "TIME",        0.70),
        ):
            for match in regex.finditer(text):
                spans.append((match.start(), match.end(), span_type, risk))

        for term, (span_type, risk) in RISK_TERMS.items():
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.I)
            for match in pattern.finditer(text):
                spans.append((match.start(), match.end(), span_type, risk))
        return spans

    def _context_trigger_spans(self, text: str, language: str) -> list[Tuple[int, int, str, float]]:
        """Return char-span hits for tokens that follow a context keyword trigger."""
        spans = []
        for pattern, span_type, risk in self._language_context_triggers(language):
            for match in pattern.finditer(text):
                spans.append((match.start(1), match.end(1), span_type, risk))
        return spans

    def _cache_regex_hit(self, tok: Token, span_type: str, risk: float, language: str) -> None:
        word = tok.text.strip().lower()
        if len(word) < 3:
            return
        if not re.search(r"[A-Za-z0-9]", word):
            return
        self.word_cache[self._cache_key(language, word)] = (span_type, risk, "regex_cache")

    def _token_overlaps_char_span(self, tok: Token, cs: int, ce: int) -> bool:
        return tok.char_start < ce and tok.char_end > cs

    def _gliner_label_to_span_type(self, label: str) -> Tuple[str, float]:
        label = label.strip().lower()
        mapping = {
            "person": ("NER_PERSON", 0.80),
            "organization": ("NER_ORG", 0.80),
            "location": ("LOCATION", 0.75),
            "username": ("SECRET", 0.90),
            "password": ("SECRET", 0.95),
            "email": ("EMAIL", 0.95),
            "phone number": ("PHONE", 0.90),
            "passport number": ("PII", 0.90),
            "driver license": ("PII", 0.90),
            "id card number": ("PII", 0.90),
            "street address": ("LOCATION", 0.80),
            "postal code": ("PII", 0.80),
            "city": ("LOCATION", 0.75),
            "state": ("LOCATION", 0.75),
            "date of birth": ("DATE", 0.85),
        }
        return mapping.get(label, ("PII", 0.75))

    def _model_char_spans(self, text: str) -> list[Tuple[int, int, str, float]]:
        if text in self.doc_span_cache:
            return self.doc_span_cache[text]
        if not self.gliner_model:
            self.doc_span_cache[text] = []
            return []
        self.stats["nlp_calls"] += 1
        try:
            entities = self.gliner_model.predict_entities(text, list(GLINER_LABELS))
        except TypeError:
            try:
                entities = self.gliner_model.predict_entities(text)
            except Exception:
                self.doc_span_cache[text] = []
                return []
        except Exception:
            self.doc_span_cache[text] = []
            return []

        spans = []
        for ent in entities:
            start = ent.get("start") or ent.get("start_char")
            end = ent.get("end") or ent.get("end_char")
            label = ent.get("label", "")
            score = float(ent.get("score", ent.get("confidence", 0.0)) or 0.0)
            if start is None or end is None or score < 0.5:
                continue
            span_type, base_risk = self._gliner_label_to_span_type(str(label))
            spans.append((int(start), int(end), span_type, max(base_risk, min(score, 0.99))))
        self.doc_span_cache[text] = spans
        return spans

    def _nlp_classify(self, tok: Token, model_hits: list) -> Tuple[str, float, str] | None:
        for cs, ce, span_type, risk in model_hits:
            if tok.char_start < ce and tok.char_end > cs:
                return (span_type, risk, "nlp_ner")
        return None

    def _classify_token(self, tok: Token, text: str, language: str, model_hits: list) -> Tuple[str, float, str] | None:
        word = tok.text.strip().lower()
        if not re.match(r"\w", word):
            return None

        cache_key = self._cache_key(language, word)
        if cache_key in self.word_cache:
            self.stats["cache_hits"] += 1
            span_type, risk, source = self.word_cache[cache_key]
            return (span_type, risk, source) if risk > 0.70 else None

        if self.stage == "regex":
            return None

        if self.stage == "regex+dict":
            self.stats["cache_misses"] += 1
            return None

        self.stats["cache_misses"] += 1
        result = self._nlp_classify(tok, model_hits)
        return result

    def classify(self, tokens: list[Token], text: str, language: Optional[str] = None) -> list[SensitiveToken]:
        language = self._detect_language(text, language)
        regex_hits = self._regex_char_spans(text)
        trigger_hits = self._context_trigger_spans(text, language) if self.stage == "full" else []
        model_hits = self._model_char_spans(text) if self.stage == "full" else []
        sensitive = []

        for tok in tokens:
            # 1. Hard regex (email, phone, SSN, IP, date, etc.)
            regex_match = None
            for cs, ce, span_type, risk in regex_hits:
                if self._token_overlaps_char_span(tok, cs, ce):
                    regex_match = (span_type, risk, "regex")
                    break
            if regex_match:
                self._cache_regex_hit(tok, regex_match[0], regex_match[1], language)
                sensitive.append(SensitiveToken(tok, *regex_match))
                continue

            if self.stage == "regex":
                continue

            # 2. Context-triggered value (password: X, username: X, etc.)
            trigger_match = None
            for cs, ce, span_type, risk in trigger_hits:
                if self._token_overlaps_char_span(tok, cs, ce):
                    trigger_match = (span_type, risk, "context_trigger")
                    break
            if trigger_match:
                sensitive.append(SensitiveToken(tok, *trigger_match))
                continue

            # 3. Suppress short ambiguous tokens (≤2 alpha chars) without a trigger —
            #    these are the main source of noise for SEX/STATE/TITLE labels.
            word = tok.text.strip()
            if self.stage == "full" and len(word) <= 2 and re.match(r"[A-Za-z]$", word):
                continue

            # 4. Word cache
            result = self._classify_token(tok, text, language, model_hits)
            if result:
                sensitive.append(SensitiveToken(tok, *result))

        return sensitive


# Maps classifier span_type labels → simulation CATEGORY_NAMES
_SPAN_TO_SIM_CATEGORY: Dict[str, str] = {
    "NER_PERSON":  "NAME",
    "NLP_CONTEXT": "NAME",
    "SSN":         "ID",
    "CREDIT_CARD": "ID",
    "PII":         "ID",
    "PHI":         "ID",
    "FINANCIAL":   "ID",
    "SECRET":      "ID",
    "IP":          "ID",
    "EMAIL":       "CONTACT",
    "PHONE":       "CONTACT",
    "NER_GPE":     "LOCATION",
    "NER_LOC":     "LOCATION",
    "LOCATION":    "LOCATION",
    "NER_ORG":     "ORG",
    "NER_DATE":    "DOB",
    "DATE":        "DOB",
    "TIME":        "DOB",
}

_SIM_CATEGORIES = ("NAME", "ID", "CONTACT", "LOCATION", "DOB", "ORG")


def classify_prompt(
    text: str,
    cache_file: str | None = None,
) -> Dict:
    """Detect sensitive tokens in a single prompt and return simulation-ready counts.

    Returns a dict with:
        n_tokens        – total token count
        n_sensitive     – number of unique sensitive tokens detected
        categories      – {NAME, ID, CONTACT, LOCATION, DOB, ORG} counts
        sensitive_tokens – list of SensitiveToken objects
    Saves the word cache to cache_file if provided.
    """
    tokenizer = BPETokenizer()
    classifier = TokenSensitivityClassifier(cache_file=cache_file)

    tokens = tokenizer.tokenize(text)
    raw_sensitive = classifier.classify(tokens, text)

    # deduplicate: one token can only be sensitive once (keep highest risk)
    by_idx: Dict[int, "SensitiveToken"] = {}
    for st in raw_sensitive:
        if st.token.idx not in by_idx or st.risk > by_idx[st.token.idx].risk:
            by_idx[st.token.idx] = st

    categories: Dict[str, int] = {cat: 0 for cat in _SIM_CATEGORIES}
    for st in by_idx.values():
        sim_cat = _SPAN_TO_SIM_CATEGORY.get(st.span_type)
        if sim_cat is None:
            sim_cat = "ID" if st.risk >= 0.85 else "NAME"
        categories[sim_cat] += 1

    classifier.save_cache()

    return {
        "n_tokens": len(tokens),
        "n_sensitive": len(by_idx),
        "categories": categories,
        "sensitive_tokens": list(by_idx.values()),
    }


def build_request_pool(
    pool_file: str,
    cache_file: str | None = None,
    mtsamples_limit: int | None = None,
    include_ai4privacy: bool = False,
    ai4privacy_limit: int = 1000,
    ai4privacy_split: str = "train",
) -> None:
    """Pre-classify all dataset rows and save a request pool JSON for the simulation.

    Each entry: {n_tokens, n_sensitive, categories, specialty}
    Load in simulation with --real-data pool.json so all experiments use real prompts.
    """
    tokenizer = BPETokenizer()
    classifier = TokenSensitivityClassifier(cache_file=cache_file)
    pool = []

    sources = list(iter_mtsamples(mtsamples_limit))
    if include_ai4privacy:
        sources += list(iter_ai4privacy(ai4privacy_limit, ai4privacy_split))

    print(f"Pre-classifying {len(sources)} prompts for request pool...")
    for idx, (specialty, text) in enumerate(sources):
        if idx % 100 == 0:
            print(f"  {idx}/{len(sources)}...", end="\r", flush=True)
        tokens = tokenizer.tokenize(text)
        raw_sensitive = classifier.classify(tokens, text)

        by_idx: Dict[int, SensitiveToken] = {}
        for st in raw_sensitive:
            if st.token.idx not in by_idx or st.risk > by_idx[st.token.idx].risk:
                by_idx[st.token.idx] = st

        categories: Dict[str, int] = {cat: 0 for cat in _SIM_CATEGORIES}
        for st in by_idx.values():
            sim_cat = _SPAN_TO_SIM_CATEGORY.get(st.span_type)
            if sim_cat is None:
                sim_cat = "ID" if st.risk >= 0.85 else "NAME"
            categories[sim_cat] += 1

        pool.append({
            "n_tokens": max(len(tokens), 1),
            "n_sensitive": len(by_idx),
            "categories": categories,
            "specialty": specialty,
        })

    classifier.save_cache()
    with open(pool_file, "w") as f:
        json.dump(pool, f)
    print(f"\nSaved {len(pool)} entries to request pool: {pool_file}")


def iter_mtsamples(rows_limit: int | None = None) -> Iterable[Tuple[str, str]]:
    df = pd.read_csv("mtsamples.csv")
    df = df.dropna(subset=["transcription"])
    if rows_limit is not None:
        df = df.head(rows_limit)
    print(f"Loaded {len(df)} MTSamples transcriptions")

    for _, row in df.iterrows():
        text = str(row["transcription"])[:1000]
        specialty = str(row["medical_specialty"]).strip().lower()
        yield specialty, text


def iter_ai4privacy(rows_limit: int | None = None, split: str = "train") -> Iterable[Tuple[str, str]]:
    if not _DATASETS_AVAILABLE:
        raise RuntimeError("datasets is not installed; cannot load ai4privacy/pii-masking-300k")

    ds = load_dataset("ai4privacy/pii-masking-300k", split=split)
    ds = ds.filter(lambda row: str(row.get("language", "")).lower() in ("english", "spanish"))

    if rows_limit is not None:
        rows_limit = min(rows_limit, len(ds))
        ds = ds.select(range(rows_limit))

    print(f"Loaded {len(ds)} AI4Privacy rows from split={split} (English + Spanish)")

    for row in ds:
        text = str(row.get("source_text", ""))[:1000]
        if not text.strip():
            continue
        yield "ai4privacy", text


def update_stats(
    stats: Dict[str, Dict[str, float]],
    bucket: str,
    text: str,
    tokenizer: BPETokenizer,
    classifier: TokenSensitivityClassifier,
) -> None:
    tokens = tokenizer.tokenize(text)
    sensitive_tokens = classifier.classify(tokens, text)
    total = len(tokens)
    sensitive = len({st.token.idx for st in sensitive_tokens})

    stats[bucket]["total_tokens"] += total
    stats[bucket]["sensitive_tokens"] += sensitive
    stats[bucket]["count"] += 1

    stats["__overall__"]["total_tokens"] += total
    stats["__overall__"]["sensitive_tokens"] += sensitive
    stats["__overall__"]["count"] += 1


def print_stats(stats: Dict[str, Dict[str, float]], classifier: TokenSensitivityClassifier) -> None:
    print(f"\n{'Bucket':<40} {'Samples':<10} {'Sens ratio':<12} {'Avg tokens'}")
    print("-" * 75)

    for bucket, values in sorted(
        ((k, v) for k, v in stats.items() if k != "__overall__"),
        key=lambda x: -x[1]["count"],
    ):
        if values["total_tokens"] == 0:
            continue
        ratio = values["sensitive_tokens"] / values["total_tokens"]
        avg_tok = values["total_tokens"] / values["count"]
        print(f"  {bucket:<38} {int(values['count']):<10} {ratio:<12.3f} {avg_tok:.0f}")

    overall = stats["__overall__"]
    overall_ratio = overall["sensitive_tokens"] / overall["total_tokens"]

    print(f"\n  OVERALL sensitive token ratio: {overall_ratio:.3f} ({overall_ratio:.1%})")
    print(
        f"  Detector cache stats: hits={classifier.stats['cache_hits']} "
        f"misses={classifier.stats['cache_misses']} "
        f"nlp_calls={classifier.stats['nlp_calls']}"
    )

    print("\n# Paste this into zkp_simpy_simulation.py:")
    print("SENSITIVE_RATIOS = {")
    print(f'    "healthcare": ({overall_ratio*0.85:.2f}, {overall_ratio*1.15:.2f}),  # calibrated from loaded datasets')
    print('    "finance":    (0.15, 0.30),   # literature estimate')
    print('    "general":    (0.05, 0.15),   # literature estimate')
    print('    "auth":       (0.30, 0.50),   # literature estimate')
    print("}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate sensitive-token ratios using the simulation detector.")
    parser.add_argument("--include-ai4privacy", action="store_true", help="Include ai4privacy/pii-masking-300k")
    parser.add_argument("--mtsamples-limit", type=int, default=None, help="Optional row cap for mtsamples.csv")
    parser.add_argument("--ai4privacy-limit", type=int, default=5000, help="Optional row cap for AI4Privacy")
    parser.add_argument("--ai4privacy-split", default="train", help="HF dataset split to load")
    parser.add_argument("--cache", default=None, metavar="PATH",
                        help="JSON file to load cached token classifications from (and save to) after calibration")
    parser.add_argument("--save-pool", default=None, metavar="PATH",
                        help="Pre-classify all dataset rows and save a request pool JSON for the simulation")
    parser.add_argument("--stage", default="full", choices=["regex", "regex+dict", "full"],
                        help="Detector stage: regex (fastest), regex+dict, or full (GLiNER)")
    args = parser.parse_args()

    if args.save_pool:
        build_request_pool(
            pool_file=args.save_pool,
            cache_file=args.cache,
            mtsamples_limit=args.mtsamples_limit,
            include_ai4privacy=args.include_ai4privacy,
            ai4privacy_limit=args.ai4privacy_limit,
            ai4privacy_split=args.ai4privacy_split,
        )
        return

    tokenizer = BPETokenizer()
    classifier = TokenSensitivityClassifier(cache_file=args.cache, stage=args.stage)
    stats = defaultdict(lambda: {"total_tokens": 0, "sensitive_tokens": 0, "count": 0})

    mtsamples_rows = list(iter_mtsamples(args.mtsamples_limit))
    mt_iter = tqdm(mtsamples_rows, desc="MTSamples (GLiNER)", unit="doc") if _TQDM_AVAILABLE else mtsamples_rows
    for bucket, text in mt_iter:
        update_stats(stats, bucket, text, tokenizer, classifier)

    if args.include_ai4privacy:
        ai4privacy_rows = list(iter_ai4privacy(args.ai4privacy_limit, args.ai4privacy_split))
        ai_iter = tqdm(ai4privacy_rows, desc="AI4Privacy (GLiNER)", unit="doc") if _TQDM_AVAILABLE else ai4privacy_rows
        for bucket, text in ai_iter:
            update_stats(stats, bucket, text, tokenizer, classifier)

    print_stats(stats, classifier)
    classifier.save_cache()


if __name__ == "__main__":
    main()
