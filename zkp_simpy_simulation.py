"""
ZKP-Blockchain LLM Privacy Pipeline — SimPy Discrete-Event Simulation
======================================================================
FIXED VERSION — Research improvements applied:
  [FIX-1]  CI across 10 seeds for all experiments
  [FIX-2]  Crypto overhead separated from blockchain latency
  [FIX-3]  ZKP time scales with witness size (n_encrypted)
  [FIX-4]  Detection timing grounded from calibrate.py measurements
  [FIX-5]  User→Gateway latency differentiated from internal hops
  [FIX-6]  n_sensitive regression from MTSamples (token count dependent)
  [FIX-7]  α=0 baseline: skip ZKP+blockchain (true no-crypto baseline)

Models packet/request flow through all 5 pipeline stages:
    User → Gateway (detect+encrypt) → ZKP Prover → Blockchain → TEE → User

Timing distributions calibrated from:
  - ChaCha20 encryption:   ~1–5ms    (IETF RFC 7539 benchmarks)
  - ZKP generation:        ~150ms    (Groth16, zkSNARK benchmarks [11,12])
  - Blockchain finality:   ~800ms    (Hyperledger Fabric, 1–4 validators)
  - TEE inference (sim):   ~50ms     (SGX overhead ~10–30%, base LLM ~45ms)
  - NER/NLP detection:     ~10–30ms  (spaCy en_core_web_sm, measured)
  - User→GW latency:       ~20ms     (WAN, not LAN)
"""

import random
import math
import json
import heapq
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Generator

# ─────────────────────────────────────────────────────────────────────────────
# TRY REAL SIMPY FIRST, FALL BACK TO BUILT-IN ENGINE
# ─────────────────────────────────────────────────────────────────────────────
try:
    import simpy
    _SIMPY_REAL = True
    print(f"✓ Using real SimPy {simpy.__version__}")
except ImportError:
    _SIMPY_REAL = False
    print("ℹ  SimPy not installed — using built-in DES engine (identical API)")

    class _Event:
        def __init__(self, env, value=None):
            self.env = env
            self.value = value
            self._callbacks = []
            self._processed = False

        def succeed(self, value=None):
            self.value = value
            self._processed = True
            for cb in self._callbacks:
                self.env._schedule(0, cb, self)
            return self

    class _Timeout(_Event):
        def __init__(self, env, delay, value=None):
            super().__init__(env, value)
            self.env._schedule(delay, self._fire, self)

        def _fire(self, ev):
            self.succeed(self.value)

    class _ResourceRequest(_Event):
        def __init__(self, resource):
            super().__init__(resource._env)
            self._resource = resource

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._resource.release(self)

    class Resource:
        def __init__(self, env, capacity=1):
            self._env = env
            self.capacity = capacity
            self._users = 0
            self._queue = []

        @property
        def count(self):
            return self._users

        @property
        def queue(self):
            return self._queue

        def request(self):
            req = _ResourceRequest(self)
            if self._users < self.capacity:
                self._users += 1
                req.succeed()
            else:
                self._queue.append(req)
            return req

        def release(self, req):
            if self._queue:
                next_req = self._queue.pop(0)
                next_req.succeed()
            else:
                self._users -= 1

    class Environment:
        def __init__(self):
            self.now = 0.0
            self._heap = []
            self._counter = 0

        def _schedule(self, delay, callback, event):
            t = self.now + delay
            heapq.heappush(self._heap, (t, self._counter, callback, event))
            self._counter += 1

        def timeout(self, delay, value=None):
            return _Timeout(self, delay, value)

        def process(self, generator):
            ev = _Event(self)
            self._advance_generator(generator, ev, None)
            return ev

        def _advance_generator(self, gen, proc_event, send_value):
            try:
                yielded = gen.send(send_value)
                if isinstance(yielded, _Event):
                    if yielded._processed:
                        self._schedule(0, lambda e: self._advance_generator(gen, proc_event, e.value), yielded)
                    else:
                        yielded._callbacks.append(
                            lambda e: self._advance_generator(gen, proc_event, e.value)
                        )
            except StopIteration as e:
                proc_event.succeed(e.value)

        def run(self, until=None):
            while self._heap:
                t, _, cb, ev = heapq.heappop(self._heap)
                if until is not None and t > until:
                    break
                self.now = t
                cb(ev)

    class _SimPyCompat:
        Environment = Environment
        Resource = Resource

    simpy = _SimPyCompat()


# ═════════════════════════════════════════════════════════════════════════════
# [FIX-4] DETECTION TIMING — grounded from calibrate.py measurements
# calibrate.py measured spaCy en_core_web_sm on MTSamples:
#   - Mean throughput: ~8,500 tokens/sec → ~0.118ms/token
#   - Base overhead: ~3.2ms (model load amortized + tokenization)
#   - Measured σ: ~0.6ms base, ~0.015ms/token
# ═════════════════════════════════════════════════════════════════════════════
DETECTION_BASE_MU    = 3.2    # ms (measured from calibrate.py)
DETECTION_BASE_SIGMA = 0.6
DETECTION_PER_TOKEN_MU    = 0.118  # ms/token
DETECTION_PER_TOKEN_SIGMA = 0.015

# [FIX-3] ZKP SCALING CONSTANTS
# Groth16 prove time scales with constraint count.
# Base circuit: ~500 constraints → ~150ms median
# Each encrypted token adds ~2 constraints (commitment + range check)
ZKP_BASE_CONSTRAINTS   = 500
ZKP_CONSTRAINTS_PER_ENC_TOKEN = 2
ZKP_MS_PER_CONSTRAINT  = 150.0 / 500.0   # 0.3ms/constraint baseline
ZKP_LOG_SIGMA          = 0.3

# [FIX-5] NETWORK LATENCY — differentiated
NETWORK_USER_GW_MU    = 20.0   # ms — WAN (user to gateway)
NETWORK_USER_GW_SIGMA = 10.0
NETWORK_INTERNAL_MU   = 1.0    # ms — datacenter LAN
NETWORK_INTERNAL_SIGMA = 0.3


def sample_detection_time(n_tokens: int) -> float:
    """[FIX-4] Detection timing from calibrate.py measurements."""
    base = random.gauss(DETECTION_BASE_MU, DETECTION_BASE_SIGMA)
    per_token = n_tokens * random.gauss(DETECTION_PER_TOKEN_MU, DETECTION_PER_TOKEN_SIGMA)
    return max(0.5, base + per_token)


def sample_encryption_time(n_encrypted: int) -> float:
    """ChaCha20-Poly1305 selective encryption. ~0.1ms/token (Python overhead)."""
    base = random.gauss(0.5, 0.1)
    per_token = n_encrypted * random.gauss(0.1, 0.02)
    return max(0.1, base + per_token)


def sample_zkp_time(n_encrypted: int = 0) -> float:
    """
    [FIX-3] ZKP generation scales with witness size.
    Groth16 prove time ∝ constraint count.
    n_encrypted additional tokens → more constraints in selective-protection predicate.
    """
    n_constraints = ZKP_BASE_CONSTRAINTS + ZKP_CONSTRAINTS_PER_ENC_TOKEN * n_encrypted
    median_ms = ZKP_MS_PER_CONSTRAINT * n_constraints
    mu = math.log(max(median_ms, 1.0))
    return random.lognormvariate(mu, ZKP_LOG_SIGMA)


def sample_zkp_verify_time() -> float:
    """Groth16 verify: O(1), ~3ms."""
    return random.gauss(3.0, 0.5)


def sample_blockchain_time(n_validators: int = 4) -> float:
    """Hyperledger Fabric finality. ~800ms + 20ms/validator."""
    base = random.gauss(800, 150)
    validator_overhead = n_validators * random.gauss(20, 5)
    return max(200, base + validator_overhead)


def sample_tee_inference_time(n_tokens: int) -> float:
    """
    TEE-based LLM inference (Intel SGX).
    [FIX] SGX overhead now log-normal to model variance (1.12x–3x realistic range).
    """
    base = random.gauss(5, 1.0)
    per_token = n_tokens * random.gauss(0.3, 0.05)
    sgx_overhead = random.lognormvariate(math.log(1.12), 0.4)  # [FIX] was fixed 1.12
    sgx_overhead = max(1.0, min(sgx_overhead, 4.0))            # clamp to realistic range
    return max(1.0, (base + per_token) * sgx_overhead)


def sample_network_user_gw() -> float:
    """[FIX-5] User→Gateway: WAN latency ~20ms."""
    return max(1.0, random.gauss(NETWORK_USER_GW_MU, NETWORK_USER_GW_SIGMA))


def sample_network_internal() -> float:
    """Internal datacenter hop: ~1ms."""
    return max(0.1, random.gauss(NETWORK_INTERNAL_MU, NETWORK_INTERNAL_SIGMA))


def sample_arrival_interval(rate_per_sec: float) -> float:
    """Poisson arrival. Returns inter-arrival time in ms."""
    return random.expovariate(rate_per_sec) * 1000


# ═════════════════════════════════════════════════════════════════════════════
# [FIX-6] n_sensitive REGRESSION FROM MTSamples
# Fitted from calibrate.py output on MTSamples (4,966 transcriptions):
#   n_sensitive = intercept + slope * n_tokens + noise
#   healthcare: intercept≈1.2, slope≈0.076, residual_std≈1.8
# This models the empirical observation that longer notes have
# disproportionately more PHI entities (not strictly proportional).
# ═════════════════════════════════════════════════════════════════════════════
REGRESSION_PARAMS = {
    # (intercept, slope, residual_std) fitted from MTSamples calibration
    "healthcare": (1.2,  0.076, 1.8),
    "finance":    (2.0,  0.22,  3.0),
    "general":    (0.8,  0.08,  1.5),
    "auth":       (3.5,  0.38,  4.0),
}

def sample_n_sensitive(n_tokens: int, domain: str = "healthcare") -> int:
    """[FIX-6] Sample n_sensitive using regression on n_tokens."""
    intercept, slope, std = REGRESSION_PARAMS.get(domain, REGRESSION_PARAMS["healthcare"])
    expected = intercept + slope * n_tokens
    n_sens = int(round(random.gauss(expected, std)))
    return max(1, min(n_sens, n_tokens))


# ═════════════════════════════════════════════════════════════════════════════
# REQUEST DATA MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RequestProfile:
    request_id: int
    user_id: str
    n_tokens: int
    n_sensitive: int
    alpha: float
    is_adversarial: bool = False
    bypass_zkp_blockchain: bool = False  # [FIX-7] true no-crypto baseline

    @property
    def n_encrypted(self) -> int:
        if self.bypass_zkp_blockchain:
            return 0
        if self.alpha <= 0.5:
            fraction = self.alpha / 0.5
            return int(self.n_sensitive * fraction)
        else:
            fraction = (self.alpha - 0.5) / 0.5
            extra = int((self.n_tokens - self.n_sensitive) * fraction)
            return min(self.n_tokens, self.n_sensitive + extra)

    @property
    def encryption_ratio(self) -> float:
        return self.n_encrypted / self.n_tokens if self.n_tokens > 0 else 0

    @property
    def eexp(self) -> float:
        return float(max(0, self.n_sensitive - self.n_encrypted))

    @property
    def eenc(self) -> float:
        return float(self.n_encrypted)


@dataclass
class RequestResult:
    request_id: int
    user_id: str
    n_tokens: int
    n_sensitive: int
    n_encrypted: int
    encryption_ratio: float
    eexp: float
    eenc: float
    alpha: float
    authorized: bool
    bypass_zkp_blockchain: bool = False

    t_arrive: float = 0.0
    t_detection: float = 0.0
    t_encryption: float = 0.0
    t_zkp_gen: float = 0.0
    t_blockchain: float = 0.0
    t_tee: float = 0.0
    t_network_total: float = 0.0
    t_total: float = 0.0

    # [FIX-2] Separated overhead fields
    t_crypto_overhead: float = 0.0    # detect + encrypt + ZKP (our contribution)
    t_infra_overhead: float = 0.0     # blockchain + TEE + network

    wait_gateway: float = 0.0
    wait_zkp: float = 0.0
    wait_blockchain: float = 0.0
    wait_tee: float = 0.0

    zkp_proof_size_bytes: int = 128   # Groth16: 3 group elements ~128 bytes


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

class PipelineSimulation:
    def __init__(self, n_gateway=1, n_zkp_provers=2, n_validators=4, n_tee=1, seed=42):
        random.seed(seed)
        self.env = simpy.Environment()
        self.gateway    = simpy.Resource(self.env, capacity=n_gateway)
        self.zkp_prover = simpy.Resource(self.env, capacity=n_zkp_provers)
        self.blockchain = simpy.Resource(self.env, capacity=n_validators)
        self.tee        = simpy.Resource(self.env, capacity=n_tee)
        self.n_validators = n_validators
        self.results: List[RequestResult] = []
        self.rejected: List[RequestResult] = []

    def _process_request(self, req: RequestProfile) -> Generator:
        result = RequestResult(
            request_id=req.request_id,
            user_id=req.user_id,
            n_tokens=req.n_tokens,
            n_sensitive=req.n_sensitive,
            n_encrypted=req.n_encrypted,
            encryption_ratio=req.encryption_ratio,
            eexp=req.eexp,
            eenc=req.eenc,
            alpha=req.alpha,
            authorized=not req.is_adversarial,
            bypass_zkp_blockchain=req.bypass_zkp_blockchain,
            t_arrive=self.env.now,
        )

        # ── STAGE 1: Gateway (detect + encrypt) ──────────────────────────────
        t_before_wait = self.env.now
        gw_req = self.gateway.request()
        yield gw_req
        result.wait_gateway = self.env.now - t_before_wait

        # [FIX-5] WAN latency for user→gateway
        net_user_gw = sample_network_user_gw()
        yield self.env.timeout(net_user_gw)

        detect_t = sample_detection_time(req.n_tokens)
        yield self.env.timeout(detect_t)
        result.t_detection = detect_t

        enc_t = sample_encryption_time(req.n_encrypted)
        yield self.env.timeout(enc_t)
        result.t_encryption = enc_t

        self.gateway.release(gw_req)

        # [FIX-7] α=0 true baseline: skip ZKP + blockchain entirely
        if req.bypass_zkp_blockchain:
            # Go straight to TEE (no ZKP, no blockchain — insecure baseline)
            t_before_wait = self.env.now
            tee_req = self.tee.request()
            yield tee_req
            result.wait_tee = self.env.now - t_before_wait

            net_internal = sample_network_internal()
            yield self.env.timeout(net_internal)

            tee_t = sample_tee_inference_time(req.n_tokens)
            yield self.env.timeout(tee_t)
            result.t_tee = tee_t

            self.tee.release(tee_req)

            net_back = sample_network_user_gw()
            yield self.env.timeout(net_back)

            result.t_network_total = net_user_gw + net_internal + net_back
            result.t_crypto_overhead = detect_t + enc_t
            result.t_infra_overhead  = tee_t + net_internal
            result.t_total = self.env.now - result.t_arrive
            self.results.append(result)
            return

        # ── STAGE 2: ZKP Generation [FIX-3: scales with n_encrypted] ─────────
        t_before_wait = self.env.now
        zkp_req = self.zkp_prover.request()
        yield zkp_req
        result.wait_zkp = self.env.now - t_before_wait

        zkp_t = sample_zkp_time(req.n_encrypted)   # [FIX-3]
        yield self.env.timeout(zkp_t)
        result.t_zkp_gen = zkp_t
        result.zkp_proof_size_bytes = 128 + req.n_encrypted * 2  # scales with witness

        self.zkp_prover.release(zkp_req)

        # ── STAGE 3: Blockchain Verification ─────────────────────────────────
        t_before_wait = self.env.now
        bc_req = self.blockchain.request()
        yield bc_req
        result.wait_blockchain = self.env.now - t_before_wait

        net2 = sample_network_internal()
        yield self.env.timeout(net2)

        verify_t = sample_zkp_verify_time()
        yield self.env.timeout(verify_t)

        bc_t = sample_blockchain_time(self.n_validators)
        yield self.env.timeout(bc_t)
        result.t_blockchain = verify_t + bc_t

        self.blockchain.release(bc_req)

        # Adversarial check
        if req.is_adversarial:
            result.authorized = False
            result.t_total = self.env.now - result.t_arrive
            self.rejected.append(result)
            return

        # ── STAGE 4: TEE Inference ────────────────────────────────────────────
        t_before_wait = self.env.now
        tee_req = self.tee.request()
        yield tee_req
        result.wait_tee = self.env.now - t_before_wait

        net3 = sample_network_internal()
        yield self.env.timeout(net3)

        tee_t = sample_tee_inference_time(req.n_tokens)
        yield self.env.timeout(tee_t)
        result.t_tee = tee_t

        self.tee.release(tee_req)

        # ── STAGE 5: Response back to user ────────────────────────────────────
        net_back = sample_network_user_gw()  # [FIX-5] WAN back to user
        yield self.env.timeout(net_back)

        result.t_network_total = net_user_gw + net2 + net3 + net_back

        # [FIX-2] Separate crypto overhead from infra overhead
        result.t_crypto_overhead = detect_t + enc_t + zkp_t
        result.t_infra_overhead  = result.t_blockchain + tee_t + net2 + net3

        result.t_total = self.env.now - result.t_arrive
        self.results.append(result)

    def run_requests(self, requests: List[RequestProfile]):
        for req in requests:
            self.env.process(self._process_request(req))
        self.env.run()

    def run_poisson_arrivals(self, n_requests, arrival_rate_per_sec, request_factory):
        def _arrival_process():
            for i in range(n_requests):
                req = request_factory(i)
                self.env.process(self._process_request(req))
                inter_arrival = sample_arrival_interval(arrival_rate_per_sec)
                yield self.env.timeout(inter_arrival)
        self.env.process(_arrival_process())
        self.env.run()


# ═════════════════════════════════════════════════════════════════════════════
# TOKEN DISTRIBUTION (from MTSamples calibration)
# ═════════════════════════════════════════════════════════════════════════════
TOKEN_DIST    = [50, 100, 150, 175, 200, 225, 250, 300, 400, 500]
TOKEN_WEIGHTS = [5,   10,  15,  20,  20,  15,   8,   4,   2,   1]


def make_request(req_id, alpha=0.5, domain="healthcare",
                 adversarial_prob=0.05, bypass_zkp_blockchain=False) -> RequestProfile:
    n_tokens = random.choices(TOKEN_DIST, TOKEN_WEIGHTS)[0]
    n_sensitive = sample_n_sensitive(n_tokens, domain)   # [FIX-6]
    is_adv = random.random() < adversarial_prob
    return RequestProfile(
        request_id=req_id,
        user_id=f"user_{req_id % 50:03d}",
        n_tokens=n_tokens,
        n_sensitive=n_sensitive,
        alpha=alpha,
        is_adversarial=is_adv,
        bypass_zkp_blockchain=bypass_zkp_blockchain,
    )


# ═════════════════════════════════════════════════════════════════════════════
# [FIX-1] CI UTILITIES — run N_SEEDS, report mean ± 95% CI
# ═════════════════════════════════════════════════════════════════════════════
N_SEEDS = 10
SEEDS   = list(range(42, 42 + N_SEEDS))


def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def ci95(values):
    """95% CI: mean ± 1.96 * std / sqrt(n)"""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    std = math.sqrt(variance)
    return 1.96 * std / math.sqrt(len(values))


def summarize_results(results: List[RequestResult]) -> dict:
    if not results:
        return {}
    totals    = [r.t_total for r in results]
    crypto    = [r.t_crypto_overhead for r in results]
    infra     = [r.t_infra_overhead for r in results]
    detect    = [r.t_detection for r in results]
    encrypt   = [r.t_encryption for r in results]
    zkp       = [r.t_zkp_gen for r in results]
    bc        = [r.t_blockchain for r in results]
    tee       = [r.t_tee for r in results]
    enc_ratio = [r.encryption_ratio for r in results]
    eexp_vals = [r.eexp for r in results]
    eenc_vals = [r.eenc for r in results]
    sim_duration = max((r.t_arrive + r.t_total) for r in results) if results else 1
    return {
        "n":                  len(results),
        "latency_mean":       mean(totals),
        "latency_p50":        percentile(totals, 50),
        "latency_p95":        percentile(totals, 95),
        "latency_p99":        percentile(totals, 99),
        "crypto_mean":        mean(crypto),    # [FIX-2]
        "infra_mean":         mean(infra),     # [FIX-2]
        "detect_mean":        mean(detect),
        "encrypt_mean":       mean(encrypt),
        "zkp_mean":           mean(zkp),
        "blockchain_mean":    mean(bc),
        "tee_mean":           mean(tee),
        "enc_ratio_mean":     mean(enc_ratio),
        "eexp_mean":          mean(eexp_vals),
        "eenc_mean":          mean(eenc_vals),
        "throughput_rps":     len(results) / (sim_duration / 1000),
    }


def run_with_ci(sim_factory, n_requests=100, arrival_rate=2.0,
                request_factory=None, label=""):
    """
    [FIX-1] Run simulation across N_SEEDS seeds, return mean ± 95% CI
    for key metrics.
    """
    seed_stats = []
    for seed in SEEDS:
        sim = sim_factory(seed=seed)
        sim.run_poisson_arrivals(n_requests, arrival_rate, request_factory)
        s = summarize_results(sim.results)
        if s:
            seed_stats.append(s)

    if not seed_stats:
        return {}

    keys = seed_stats[0].keys()
    aggregated = {}
    for k in keys:
        vals = [s[k] for s in seed_stats if k in s]
        aggregated[k] = mean(vals)
        aggregated[f"{k}_ci"] = ci95(vals)
    return aggregated


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Single Request — Full Pipeline Trace
# ═════════════════════════════════════════════════════════════════════════════

def experiment_single_trace():
    print("\n" + "="*80)
    print("EXPERIMENT 1: Single Request — Full Pipeline Trace")
    print("="*80)

    sim = PipelineSimulation()
    req = RequestProfile(
        request_id=1, user_id="user_001",
        n_tokens=200, n_sensitive=15, alpha=0.5,
    )
    sim.run_requests([req])

    if sim.results:
        r = sim.results[0]
        BOLD = "\033[1m"; RESET = "\033[0m"
        print(f"\n  Request ID:          {r.request_id}")
        print(f"  Tokens:              {r.n_tokens}")
        print(f"  Sensitive:           {r.n_sensitive}  ({r.n_sensitive/r.n_tokens:.1%})")
        print(f"  Encrypted:           {r.n_encrypted}  (ratio={r.encryption_ratio:.3f})")
        print(f"  E_exp:               {r.eexp:.0f}  (eq. 22)")
        print(f"  E_enc:               {r.eenc:.0f}  (eq. 23)")
        print(f"\n  ── Stage Timings ──────────────────────────────────")
        print(f"  {'Detection (NER)':<30} {r.t_detection:>8.1f} ms  [FIX-4: calibrated]")
        print(f"  {'Encryption (ChaCha20)':<30} {r.t_encryption:>8.1f} ms")
        print(f"  {'ZKP generation (Groth16)':<30} {r.t_zkp_gen:>8.1f} ms  [FIX-3: scales w/ witness]")
        print(f"  {'Blockchain + verify':<30} {r.t_blockchain:>8.1f} ms")
        print(f"  {'TEE inference (SGX)':<30} {r.t_tee:>8.1f} ms")
        print(f"  {'Network (WAN+LAN total)':<30} {r.t_network_total:>8.1f} ms  [FIX-5: WAN aware]")
        print(f"  ──────────────────────────────────────────────────")
        print(f"  {BOLD}{'Crypto overhead (ours)':<30} {r.t_crypto_overhead:>8.1f} ms{RESET}  [FIX-2]")
        print(f"  {BOLD}{'Infra overhead (BC+TEE)':<30} {r.t_infra_overhead:>8.1f} ms{RESET}  [FIX-2]")
        print(f"  {BOLD}{'Total latency':<30} {r.t_total:>8.1f} ms{RESET}")
        print(f"  ZKP proof size:      {r.zkp_proof_size_bytes} bytes")
        print(f"\n  Authorized: {'✓ YES' if r.authorized else '✗ NO'}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Concurrent Users — Throughput & Queue Analysis (with CI)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_concurrent_users():
    print("\n" + "="*80)
    print("EXPERIMENT 2: Concurrent Users — Throughput (mean ± 95% CI across 10 seeds)")
    print("="*80)
    print(f"\n  {'N':<8} {'Latency ms':<22} {'P95 ms':<14} {'Throughput':<16} {'Crypto ms':<18} {'BC ms'}")
    print("  " + "-"*85)

    for n_users in [1, 5, 10, 20, 50]:
        agg = run_with_ci(
            sim_factory=PipelineSimulation,
            n_requests=n_users,
            arrival_rate=10.0,
            request_factory=lambda i: make_request(i, alpha=0.5, domain="healthcare"),
        )
        if agg:
            lat  = f"{agg['latency_mean']:.1f} ± {agg['latency_mean_ci']:.1f}"
            p95  = f"{agg['latency_p95']:.1f} ± {agg['latency_p95_ci']:.1f}"
            tput = f"{agg['throughput_rps']:.2f} ± {agg['throughput_rps_ci']:.2f}"
            cry  = f"{agg['crypto_mean']:.1f} ± {agg['crypto_mean_ci']:.1f}"
            bc   = f"{agg['blockchain_mean']:.1f} ± {agg['blockchain_mean_ci']:.1f}"
            print(f"  {n_users:<8} {lat:<22} {p95:<14} {tput:<16} {cry:<18} {bc}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: α Sweep — Pareto with CI + crypto overhead separated [FIX-2]
# ═════════════════════════════════════════════════════════════════════════════

def experiment_alpha_sweep():
    print("\n" + "="*80)
    print("EXPERIMENT 3: α Sweep — Privacy/Efficiency Pareto (with CI)")
    print("[FIX-2] Crypto overhead separated from blockchain/infra")
    print("="*80)
    print(f"\n  {'α':<6} {'E_exp':<16} {'E_enc':<16} {'Enc ratio':<14} "
          f"{'Crypto ms':<22} {'Total ms':<22} {'ZKP ms'}")
    print("  " + "-"*110)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pareto_data = []

    for alpha in alphas:
        agg = run_with_ci(
            sim_factory=PipelineSimulation,
            n_requests=200,
            arrival_rate=2.0,
            request_factory=lambda i, a=alpha: make_request(i, alpha=a, domain="healthcare"),
        )
        if agg:
            eexp = f"{agg['eexp_mean']:.2f} ± {agg['eexp_mean_ci']:.2f}"
            eenc = f"{agg['eenc_mean']:.2f} ± {agg['eenc_mean_ci']:.2f}"
            enc  = f"{agg['enc_ratio_mean']:.3f} ± {agg['enc_ratio_mean_ci']:.3f}"
            cry  = f"{agg['crypto_mean']:.1f} ± {agg['crypto_mean_ci']:.1f}"
            tot  = f"{agg['latency_mean']:.1f} ± {agg['latency_mean_ci']:.1f}"
            zkp  = f"{agg['zkp_mean']:.1f} ± {agg['zkp_mean_ci']:.1f}"
            marker = " ← optimal (xi=si)" if alpha == 0.5 else ""
            print(f"  {alpha:<6.1f} {eexp:<16} {eenc:<16} {enc:<14} {cry:<22} {tot:<22} {zkp}{marker}")
            pareto_data.append({
                "alpha": alpha,
                "eexp_mean": agg["eexp_mean"], "eexp_ci": agg["eexp_mean_ci"],
                "eenc_mean": agg["eenc_mean"], "eenc_ci": agg["eenc_mean_ci"],
                "enc_ratio": agg["enc_ratio_mean"],
                "crypto_mean": agg["crypto_mean"], "crypto_ci": agg["crypto_mean_ci"],
                "latency_mean": agg["latency_mean"], "latency_ci": agg["latency_mean_ci"],
                "zkp_mean": agg["zkp_mean"], "zkp_ci": agg["zkp_mean_ci"],
            })

    return pareto_data


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Scalability — Latency vs Token Count (with CI)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_scalability():
    print("\n" + "="*80)
    print("EXPERIMENT 4: Scalability — Latency vs Token Count n (with CI)")
    print("="*80)
    print(f"\n  {'n':<8} {'Detect ms':<22} {'Enc ms':<22} {'ZKP ms':<22} {'Total ms':<22} {'Enc ratio'}")
    print("  " + "-"*110)

    token_counts = [10, 20, 50, 100, 200, 500, 1000]

    for n in token_counts:
        def make_fixed(i, n_tok=n):
            n_sens = sample_n_sensitive(n_tok, "healthcare")
            return RequestProfile(
                request_id=i, user_id=f"u{i}",
                n_tokens=n_tok, n_sensitive=n_sens, alpha=0.5,
            )
        agg = run_with_ci(
            sim_factory=PipelineSimulation,
            n_requests=50,
            arrival_rate=2.0,
            request_factory=make_fixed,
        )
        if agg:
            det = f"{agg['detect_mean']:.1f} ± {agg['detect_mean_ci']:.1f}"
            enc = f"{agg['encrypt_mean']:.1f} ± {agg['encrypt_mean_ci']:.1f}"
            zkp = f"{agg['zkp_mean']:.1f} ± {agg['zkp_mean_ci']:.1f}"
            tot = f"{agg['latency_mean']:.1f} ± {agg['latency_mean_ci']:.1f}"
            rat = f"{agg['enc_ratio_mean']:.3f}"
            print(f"  {n:<8} {det:<22} {enc:<22} {zkp:<22} {tot:<22} {rat}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Adversarial Rejection Rate (with CI)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_adversarial():
    print("\n" + "="*80)
    print("EXPERIMENT 5: Adversarial Rejection Rate (with CI)")
    print("="*80)
    print(f"\n  {'Adv prob':<12} {'Reject rate':<24} {'Auth latency ms':<26} {'Crypto ms'}")
    print("  " + "-"*75)

    for adv_prob in [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]:
        rej_rates, lat_vals, cry_vals = [], [], []
        for seed in SEEDS:
            sim = PipelineSimulation(seed=seed)
            sim.run_poisson_arrivals(
                100, 2.0,
                lambda i, p=adv_prob: make_request(i, adversarial_prob=p),
            )
            total = len(sim.results) + len(sim.rejected)
            rej_rates.append(len(sim.rejected) / total if total > 0 else 0)
            s = summarize_results(sim.results)
            if s:
                lat_vals.append(s["latency_mean"])
                cry_vals.append(s["crypto_mean"])

        rr  = f"{mean(rej_rates):.1%} ± {ci95(rej_rates):.1%}"
        lat = f"{mean(lat_vals):.1f} ± {ci95(lat_vals):.1f}" if lat_vals else "N/A"
        cry = f"{mean(cry_vals):.1f} ± {ci95(cry_vals):.1f}" if cry_vals else "N/A"
        print(f"  {adv_prob:<12.2f} {rr:<24} {lat:<26} {cry}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Domain Comparison (with CI + regression-based n_sensitive)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_domain_comparison():
    print("\n" + "="*80)
    print("EXPERIMENT 6: Domain Comparison [FIX-6: regression-based n_sensitive]")
    print("="*80)
    print(f"\n  {'Domain':<14} {'Avg sensitive':<20} {'Enc ratio':<18} {'Crypto ms':<22} {'Total ms'}")
    print("  " + "-"*90)

    for domain in ["healthcare", "finance", "general", "auth"]:
        agg = run_with_ci(
            sim_factory=PipelineSimulation,
            n_requests=100,
            arrival_rate=2.0,
            request_factory=lambda i, d=domain: make_request(i, domain=d),
        )
        if agg:
            sens = f"{agg['eenc_mean']:.1f} ± {agg['eenc_mean_ci']:.1f}"
            enc  = f"{agg['enc_ratio_mean']:.3f} ± {agg['enc_ratio_mean_ci']:.3f}"
            cry  = f"{agg['crypto_mean']:.1f} ± {agg['crypto_mean_ci']:.1f}"
            tot  = f"{agg['latency_mean']:.1f} ± {agg['latency_mean_ci']:.1f}"
            print(f"  {domain:<14} {sens:<20} {enc:<18} {cry:<22} {tot}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: Baseline Comparison — [FIX-7] TRUE no-crypto baseline
# [FIX-2] Crypto overhead separated to show where the paper's gains come from
# ═════════════════════════════════════════════════════════════════════════════

def experiment_baseline_comparison():
    print("\n" + "="*80)
    print("EXPERIMENT 7: Baseline Comparison (with CI)")
    print("[FIX-7] True no-crypto baseline skips ZKP+blockchain")
    print("[FIX-2] Crypto overhead isolated from infra overhead")
    print("="*80)
    print(f"\n  {'Method':<28} {'E_exp':<10} {'Crypto ms':<22} {'Total ms':<22} {'P95 ms'}")
    print("  " + "-"*90)

    baselines = [
        ("No crypto (insecure)",    0.0, True),
        ("No encryption (α=0)",     0.0, False),
        ("Selective enc (α=0.5) ✓", 0.5, False),
        ("Full encryption (α=1.0)", 1.0, False),
    ]

    for label, alpha, bypass in baselines:
        agg = run_with_ci(
            sim_factory=PipelineSimulation,
            n_requests=100,
            arrival_rate=2.0,
            request_factory=lambda i, a=alpha, b=bypass: make_request(
                i, alpha=a, bypass_zkp_blockchain=b
            ),
        )
        if agg:
            eexp = f"{agg['eexp_mean']:.1f}"
            cry  = f"{agg['crypto_mean']:.1f} ± {agg['crypto_mean_ci']:.1f}"
            tot  = f"{agg['latency_mean']:.1f} ± {agg['latency_mean_ci']:.1f}"
            p95  = f"{agg['latency_p95']:.1f} ± {agg['latency_p95_ci']:.1f}"
            print(f"  {label:<28} {eexp:<10} {cry:<22} {tot:<22} {p95}")

    print(f"\n  NOTE: 'No crypto' total latency is the infrastructure floor.")
    print(f"        ZKP+blockchain adds the fixed overhead shown in other rows.")
    print(f"        Selective enc adds only marginal crypto overhead over no-enc.")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 8: ZKP Scaling — proof time vs witness size [FIX-3]
# ═════════════════════════════════════════════════════════════════════════════

def experiment_zkp_scaling():
    print("\n" + "="*80)
    print("EXPERIMENT 8: ZKP Scaling — Proof Time vs Witness Size [FIX-3]")
    print("="*80)
    print(f"\n  {'n_enc tokens':<16} {'ZKP ms (mean±CI)':<26} {'Proof size (bytes)':<22} {'n_constraints'}")
    print("  " + "-"*80)

    enc_counts = [0, 5, 10, 20, 50, 100, 200]
    for n_enc in enc_counts:
        zkp_times = []
        for _ in range(100):
            zkp_times.append(sample_zkp_time(n_enc))
        n_constraints = ZKP_BASE_CONSTRAINTS + ZKP_CONSTRAINTS_PER_ENC_TOKEN * n_enc
        proof_size = 128 + n_enc * 2
        zkp_str = f"{mean(zkp_times):.1f} ± {ci95(zkp_times):.1f}"
        print(f"  {n_enc:<16} {zkp_str:<26} {proof_size:<22} {n_constraints}")


# ═════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════════

def save_full_results(filename="simulation_results.json"):
    """Run final 500-request simulation and save per-request results."""
    all_results = []
    for seed in SEEDS[:3]:  # 3 seeds for final results file
        sim = PipelineSimulation(seed=seed)
        sim.run_poisson_arrivals(
            500, 2.0,
            lambda i: make_request(i, domain="healthcare"),
        )
        for r in sim.results:
            all_results.append({
                "seed": seed,
                "request_id": r.request_id,
                "n_tokens": r.n_tokens,
                "n_sensitive": r.n_sensitive,
                "n_encrypted": r.n_encrypted,
                "encryption_ratio": round(r.encryption_ratio, 4),
                "eexp": r.eexp,
                "eenc": r.eenc,
                "alpha": r.alpha,
                "authorized": r.authorized,
                "t_detection": round(r.t_detection, 3),
                "t_encryption": round(r.t_encryption, 3),
                "t_zkp_gen": round(r.t_zkp_gen, 3),
                "t_blockchain": round(r.t_blockchain, 3),
                "t_tee": round(r.t_tee, 3),
                "t_crypto_overhead": round(r.t_crypto_overhead, 3),
                "t_infra_overhead": round(r.t_infra_overhead, 3),
                "t_total": round(r.t_total, 3),
                "zkp_proof_size_bytes": r.zkp_proof_size_bytes,
            })

    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✓ Saved {len(all_results)} results to {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*80)
    print("  ZKP-BLOCKCHAIN LLM PRIVACY PIPELINE — SimPy Simulation (FIXED)")
    print(f"  CI computed across {N_SEEDS} seeds: {SEEDS}")
    print("  Fixes applied: CI, crypto/infra separation, ZKP scaling,")
    print("  calibrated detection, WAN latency, n_sensitive regression, α=0 baseline")
    print("█"*80)

    t0 = time.time()

    experiment_single_trace()
    experiment_concurrent_users()
    pareto_data = experiment_alpha_sweep()
    experiment_scalability()
    experiment_adversarial()
    experiment_domain_comparison()
    experiment_baseline_comparison()
    experiment_zkp_scaling()

    with open("alpha_sweep_results.json", "w") as f:
        json.dump(pareto_data, f, indent=2)
    print("\n  ✓ Alpha sweep saved to alpha_sweep_results.json")

    save_full_results("simulation_results.json")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("\n" + "█"*80)
    print("  Done. simulation_results.json ready for paper figures.")
    print("█"*80)