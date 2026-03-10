import pandas as pd
import spacy
from collections import defaultdict

# ── Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("mtsamples.csv")
df = df.dropna(subset=["transcription"])
print(f"Loaded {len(df)} transcriptions")

nlp = spacy.load("en_core_web_sm")

# PHI-relevant NER labels
PHI_LABELS = {"PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", "CARDINAL", "LOC"}

# ── Per-specialty calibration ─────────────────────────────────────────────
specialty_stats = defaultdict(lambda: {"total_tokens": 0, "sensitive_tokens": 0, "count": 0})

for _, row in df.iterrows():
    text = str(row["transcription"])[:1000]  # cap at 1000 chars for speed
    specialty = str(row["medical_specialty"]).strip().lower()

    doc = nlp(text)
    total = len(doc)
    sensitive = sum(1 for tok in doc if any(tok.i >= ent.start and tok.i < ent.end 
                                             for ent in doc.ents if ent.label_ in PHI_LABELS))
    
    specialty_stats[specialty]["total_tokens"] += total
    specialty_stats[specialty]["sensitive_tokens"] += sensitive
    specialty_stats[specialty]["count"] += 1

# ── Print results ─────────────────────────────────────────────────────────
print(f"\n{'Specialty':<40} {'Samples':<10} {'Sens ratio':<12} {'Avg tokens'}")
print("-" * 75)

overall_total, overall_sensitive = 0, 0

for spec, s in sorted(specialty_stats.items(), key=lambda x: -x[1]["count"]):
    if s["total_tokens"] == 0:
        continue
    ratio = s["sensitive_tokens"] / s["total_tokens"]
    avg_tok = s["total_tokens"] / s["count"]
    overall_total += s["total_tokens"]
    overall_sensitive += s["sensitive_tokens"]
    print(f"  {spec:<38} {s['count']:<10} {ratio:<12.3f} {avg_tok:.0f}")

overall_ratio = overall_sensitive / overall_total
print(f"\n  OVERALL sensitive token ratio: {overall_ratio:.3f} ({overall_ratio:.1%})")

# ── Output calibrated SENSITIVE_RATIOS for your sim ──────────────────────
print("\n# Paste this into zkp_simpy_simulation.py:")
print(f'SENSITIVE_RATIOS = {{')
print(f'    "healthcare": ({overall_ratio*0.85:.2f}, {overall_ratio*1.15:.2f}),  # measured from MTSamples')
print(f'    "finance":    (0.15, 0.30),   # literature estimate')
print(f'    "general":    (0.05, 0.15),   # literature estimate')
print(f'    "auth":       (0.30, 0.50),   # literature estimate')
print(f'}}')