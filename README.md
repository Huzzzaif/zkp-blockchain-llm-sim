# ZKP-Blockchain LLM Privacy Pipeline — Simulation

Discrete-event simulation (SimPy) of a privacy-preserving LLM inference architecture that combines **selective token encryption**, **zero-knowledge proofs (ZKPs)**, and a **permissioned blockchain** to protect sensitive prompts (PII/PHI) before they reach a cloud-hosted LLM.

> Companion code for: *"Privacy-Preserving LLM Inference via Selective Encryption, ZKP-Based Policy Enforcement, and Blockchain Authorization"* — Huzaif Khan, Ghazal Mahdian, Ali Jalooli, CSUDH.

---

## Architecture

```
User → Trusted Gateway (detect + encrypt) → ZKP Prover → Blockchain → TEE (LLM) → User
```

| Component | Role |
|-----------|------|
| **Trusted Gateway** | Tokenizes prompt, classifies sensitive tokens (regex → dict → NER), applies ChaCha20-Poly1305 selective encryption |
| **ZKP Prover** | Generates Groth16 proof attesting to policy compliance (auth, domain, rate-limit, selective protection) |
| **Permissioned Blockchain** | Verifies ZKP, logs tamper-evident authorization record, enforces BFT consensus |
| **TEE (Cloud)** | Decrypts tokens inside enclave, runs LLM inference, signs response |

---

## Repository Structure

```
zkp-blockchain-llm-sim/
├── zkp_simpy_simulation.py   # Main simulation — all 7 experiments
├── calibrate.py              # Calibrates PHI token ratio from MTSamples dataset
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT
└── README.md
```

---

## Experiments

| # | Name | What it measures |
|---|------|-----------------|
| 1 | Single Request Trace | Full pipeline stage-by-stage latency |
| 2 | Concurrent Users | Throughput and queue build-up |
| 3 | α Sweep (Pareto) | Privacy–efficiency trade-off (E_exp vs E_enc) |
| 4 | Scalability | Latency vs token count n |
| 5 | Adversarial Rejection | ZKP domain predicate rejection rate |
| 6 | Domain Comparison | healthcare / finance / general / auth |
| 7 | Baseline Comparison | No encryption vs full encryption vs selective (ours) |

---

## Timing Model

All stage timings are calibrated from published benchmarks:

| Stage | Distribution | Source |
|-------|-------------|--------|
| Detection (NER) | `N(2.0, 0.5) + 0.08n ms` | spaCy en_core_web_sm |
| Encryption (ChaCha20) | `N(0.5, 0.1) + 0.1·ne ms` | RFC 7539 |
| ZKP generation (Groth16) | `LogN(ln 150, 0.3) ms` | zkSNARK benchmarks |
| Blockchain (Hyperledger) | `N(800, 150) + 20v ms` | Hyperledger Fabric docs |
| TEE inference (SGX) | `1.12 × (N(5,1) + 0.3n) ms` | Intel SGX whitepapers |

---

## Setup

```bash
# Clone
git clone https://github.com/Huzzzaif/zkp-blockchain-llm-sim.git
cd zkp-blockchain-llm-sim

# Create virtual environment
python3 -m venv zkp_env
source zkp_env/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Usage

### Run all experiments
```bash
python zkp_simpy_simulation.py
```

Outputs:
- `simulation_results.json` — full per-request results (500 requests)
- `alpha_sweep_results.json` — Pareto data for α ∈ [0, 1]

### Calibrate PHI ratio from MTSamples
```bash
# Requires mtsamples.csv in the working directory (not committed — see note below)
python calibrate.py
```

> **Note:** `mtsamples.csv` is excluded from version control (too large). Download from [Kaggle: MTSamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions).

---

## Key Parameters (Table III)

| Parameter | Value |
|-----------|-------|
| Requests (N) | 500 |
| Arrival rate (λ) | 2 req/s |
| Token range | 50–500 (weighted) |
| PHI ratio (ρ) | 7.2% (calibrated from MTSamples) |
| α (default) | 0.5 |
| ZKP circuit | ~500 constraints |
| SGX overhead | 12% |
| Blockchain validators | 4 (BFT: f < 1) |

---

## SimPy Dependency

The simulation runs with or without SimPy installed:
- **With SimPy:** uses real SimPy 4.x engine
- **Without SimPy:** falls back to a built-in minimal DES engine with identical API

```bash
pip install simpy  # optional but recommended
```

---

## License

MIT — see [LICENSE](LICENSE)v
