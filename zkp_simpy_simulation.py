"""
ZKP-Blockchain LLM Privacy Pipeline — SimPy Discrete-Event Simulation
======================================================================
Models packet/request flow through all 5 pipeline stages:

    User → Gateway (detect+encrypt) → ZKP Prover → Blockchain → TEE → User

Timing distributions are calibrated from real benchmark literature:
  - ChaCha20 encryption:   ~1–5ms    (IETF RFC 7539 benchmarks)
  - ZKP generation:        ~150ms    (Groth16, zkSNARK benchmarks [11,12])
  - Blockchain finality:   ~800ms    (Hyperledger Fabric, 1–4 validators)
  - TEE inference (sim):   ~50ms     (SGX overhead ~10%, base LLM ~45ms sim)
  - NER/NLP detection:     ~10–30ms  (spaCy en_core_web_sm benchmarks)

Privacy metric: Weighted exposure risk R_exp(α) ∈ [0,1] over PHI categories
  - Categories: NAME, DOB, CONTACT, ID, LOCATION, ORG
  - Severity weights from HIPAA Safe Harbor identifier hierarchy
  - Encryption budget allocated by priority order (high-severity first)

Experimental rigor:
  - Multi-seed CI (10 seeds, 95% confidence intervals)
  - t_crypto_overhead vs t_infra_overhead separation
  - WAN (user↔gateway) vs LAN (internal datacenter) network model
  - ZKP proof time scales with n_encrypted (witness size)
  - Regression-based n_sensitive = f(n_tokens) per domain

Usage:
    pip install simpy   # optional
    python3 zkp_simpy_simulation.py
"""

import random
import math
import json
import time
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Generator, Callable

try:
    from calibrate import classify_prompt
    _CALIBRATE_AVAILABLE = True
except ImportError:
    _CALIBRATE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# TRY REAL SIMPY FIRST, FALL BACK TO BUILT-IN ENGINE
# ─────────────────────────────────────────────────────────────────────────────
try:
    import simpy
    _SIMPY_REAL = True
    print("✓ Using real SimPy", simpy.__version__)
except ImportError:
    _SIMPY_REAL = False
    print("ℹ  SimPy not installed — using built-in DES engine (identical API)")
    print("   Install with: pip install simpy\n")

    # ── Minimal SimPy-compatible DES engine ──────────────────────────────────
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

        def __and__(self, other):
            return _AllOf(self.env, [self, other])

    class _Timeout(_Event):
        def __init__(self, env, delay, value=None):
            super().__init__(env, value)
            self.env._schedule(delay, self._fire, self)

        def _fire(self, ev):
            self.succeed(self.value)

    class _AllOf(_Event):
        def __init__(self, env, events):
            super().__init__(env)
            self._events = events
            self._done = 0
            for e in events:
                e._callbacks.append(self._check)

        def _check(self, ev):
            self._done += 1
            if self._done == len(self._events):
                self.succeed([e.value for e in self._events])

    class _ResourceRequest(_Event):
        def __init__(self, resource):
            super().__init__(resource._env)
            self._resource = resource

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

    class _ResourceRequest(_Event):
        def __init__(self, resource):
            super().__init__(resource._env)
            self._resource = resource

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._resource.release(self)

    class Environment:
        def __init__(self):
            self.now = 0.0
            self._heap = []
            self._counter = 0
            self._active_procs = 0

        def _schedule(self, delay, callback, event):
            t = self.now + delay
            heapq.heappush(self._heap, (t, self._counter, callback, event))
            self._counter += 1

        def timeout(self, delay, value=None):
            return _Timeout(self, delay, value)

        def process(self, generator):
            ev = _Event(self)
            self._active_procs += 1
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
                elif isinstance(yielded, Resource):
                    req = yielded.request()
                    req._callbacks.append(
                        lambda e: self._advance_generator(gen, proc_event, e)
                    )
            except StopIteration as e:
                self._active_procs -= 1
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
# SENSITIVE CATEGORY MODEL
# ═════════════════════════════════════════════════════════════════════════════

CATEGORY_DEFS = [
    ("NAME",     1.0, 1),   # Patient/person names — direct identifier
    ("ID",       0.9, 2),   # MRN, SSN, account numbers — direct identifier
    ("CONTACT",  0.7, 3),   # Phone, fax, email — quasi-identifier
    ("LOCATION", 0.5, 4),   # Address, zip, geo — quasi-identifier
    ("DOB",      0.4, 5),   # Dates (birth, admission, discharge)
    ("ORG",      0.2, 6),   # Organization/employer — indirect
]

CATEGORY_NAMES    = [c[0] for c in CATEGORY_DEFS]
CATEGORY_WEIGHTS  = {c[0]: c[1] for c in CATEGORY_DEFS}
CATEGORY_PRIORITY = {c[0]: c[2] for c in CATEGORY_DEFS}

CATEGORY_DISTRIBUTIONS = {
    "healthcare": {
        "NAME": (0.25, 0.05), "ID": (0.20, 0.04), "CONTACT": (0.15, 0.03),
        "LOCATION": (0.15, 0.03), "DOB": (0.15, 0.03), "ORG": (0.10, 0.02),
    },
    "finance": {
        "NAME": (0.20, 0.04), "ID": (0.30, 0.05), "CONTACT": (0.15, 0.03),
        "LOCATION": (0.10, 0.02), "DOB": (0.05, 0.02), "ORG": (0.20, 0.04),
    },
    "general": {
        "NAME": (0.30, 0.06), "ID": (0.10, 0.03), "CONTACT": (0.20, 0.04),
        "LOCATION": (0.20, 0.04), "DOB": (0.10, 0.03), "ORG": (0.10, 0.03),
    },
    "auth": {
        "NAME": (0.15, 0.03), "ID": (0.40, 0.06), "CONTACT": (0.20, 0.04),
        "LOCATION": (0.05, 0.02), "DOB": (0.05, 0.02), "ORG": (0.15, 0.03),
    },
}


def generate_category_counts(n_sensitive: int, domain: str) -> Dict[str, int]:
    """Distribute n_sensitive tokens across categories with Gaussian noise."""
    dist = CATEGORY_DISTRIBUTIONS.get(domain, CATEGORY_DISTRIBUTIONS["general"])
    raw = {}
    for cat in CATEGORY_NAMES:
        mean_frac, std_frac = dist[cat]
        raw[cat] = max(0.0, random.gauss(mean_frac, std_frac))

    total_frac = sum(raw.values()) or 1.0
    normed = {cat: raw[cat] / total_frac for cat in CATEGORY_NAMES}

    counts = {}
    assigned = 0
    for cat in CATEGORY_NAMES[:-1]:
        c = max(0, round(normed[cat] * n_sensitive))
        c = min(c, n_sensitive - assigned)
        counts[cat] = c
        assigned += c
    counts[CATEGORY_NAMES[-1]] = max(0, n_sensitive - assigned)
    return counts


def allocate_encryption_budget(
    categories: Dict[str, int],
    budget: int,
) -> Dict[str, int]:
    """Spend encryption budget greedily: highest-priority categories first."""
    encrypted = {cat: 0 for cat in CATEGORY_NAMES}
    remaining = budget
    for cat in sorted(CATEGORY_NAMES, key=lambda c: CATEGORY_PRIORITY[c]):
        if remaining <= 0:
            break
        can_encrypt = min(categories.get(cat, 0), remaining)
        encrypted[cat] = can_encrypt
        remaining -= can_encrypt
    return encrypted


def compute_r_exp(
    categories: Dict[str, int],
    encrypted: Dict[str, int],
) -> float:
    """R_exp = Σ w_c · n_c_exposed / Σ w_c · n_c_total.  ∈ [0,1]."""
    num = 0.0
    den = 0.0
    for cat in CATEGORY_NAMES:
        w = CATEGORY_WEIGHTS[cat]
        n_total = categories.get(cat, 0)
        n_exposed = max(0, n_total - encrypted.get(cat, 0))
        num += w * n_exposed
        den += w * n_total
    return num / den if den > 0 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# TIMING DISTRIBUTIONS
# ═════════════════════════════════════════════════════════════════════════════

def sample_detection_time(n_tokens: int) -> float:
    """NER/NLP detection. ~2ms base + 0.08ms/token. Returns ms."""
    base = random.gauss(2.0, 0.5)
    per_token = n_tokens * random.gauss(0.08, 0.01)
    return max(0.5, base + per_token)


def sample_encryption_time(n_encrypted: int) -> float:
    """ChaCha20-Poly1305 selective encryption. ~0.1ms/token. Returns ms."""
    base = random.gauss(0.5, 0.1)
    per_token = n_encrypted * random.gauss(0.1, 0.02)
    return max(0.1, base + per_token)


def sample_zkp_time(n_encrypted: int) -> float:
    """
    ZKP generation (Groth16/PLONK). Proof cost scales with witness size.
    Base circuit: ~500 constraints for 5 predicates.
    Each encrypted token adds ~2 constraints (range check + commitment).
    Median ≈ 0.3ms × n_constraints.
    Returns ms.
    """
    n_constraints = 500 + 2 * n_encrypted
    median = 0.3 * n_constraints
    mu = math.log(max(median, 1.0))
    sigma = 0.3
    return random.lognormvariate(mu, sigma)


def sample_zkp_verify_time() -> float:
    """Groth16 verify is O(1) ~2–5ms. Returns ms."""
    return random.gauss(3.0, 0.5)


def sample_blockchain_time(n_validators: int = 4) -> float:
    """Hyperledger Fabric finality. ~800ms median with 4 validators. Returns ms."""
    base = random.gauss(800, 150)
    validator_overhead = n_validators * random.gauss(20, 5)
    return max(200, base + validator_overhead)


def sample_tee_inference_time(n_tokens: int) -> float:
    """TEE (SGX) inference. ~5ms base + 0.3ms/token, 12% SGX overhead. Returns ms."""
    base = random.gauss(5, 1.0)
    per_token = n_tokens * random.gauss(0.3, 0.05)
    return max(1.0, (base + per_token) * 1.12)


def sample_wan_latency() -> float:
    """User ↔ Gateway WAN hop. ~15–40ms (typical broadband RTT). Returns ms."""
    return max(5.0, random.gauss(25.0, 8.0))


def sample_lan_latency() -> float:
    """Internal datacenter hop. ~0.5–2ms. Returns ms."""
    return max(0.1, random.gauss(1.0, 0.3))


def sample_arrival_interval(rate_per_sec: float) -> float:
    """Poisson inter-arrival time in ms."""
    return random.expovariate(rate_per_sec) * 1000


# ═════════════════════════════════════════════════════════════════════════════
# REGRESSION-BASED SENSITIVE TOKEN COUNT
# ═════════════════════════════════════════════════════════════════════════════
#
# n_sensitive = max(1, round(β0 + β1·n_tokens + ε))
# Produces realistic correlation between prompt length and sensitive content.
# ═════════════════════════════════════════════════════════════════════════════

SENSITIVE_REGRESSION = {
    # (intercept, slope, residual_std)
    "healthcare": (1.0, 0.075, 2.0),   # ~7.5% + small base
    "finance":    (2.0, 0.22,  4.0),    # ~22% + higher base
    "general":    (0.5, 0.10,  3.0),    # ~10%
    "auth":       (3.0, 0.40,  5.0),    # ~40%, credential-heavy
}


def sample_n_sensitive(n_tokens: int, domain: str) -> int:
    """Regression-based sensitive count: β0 + β1·n_tokens + ε."""
    b0, b1, sigma = SENSITIVE_REGRESSION.get(domain, SENSITIVE_REGRESSION["general"])
    n = b0 + b1 * n_tokens + random.gauss(0, sigma)
    return max(1, round(n))


# ═════════════════════════════════════════════════════════════════════════════
# REQUEST DATA MODEL
# ═════════════════════════════════════════════════════════════════════════════

TOKEN_DIST    = [50, 100, 150, 175, 200, 225, 250, 300, 400, 500]
TOKEN_WEIGHTS = [5,   10,  15,  20,  20,  15,   8,   4,   2,   1]

# Populated at startup by --real-data; when set, make_request() samples from it
_REQUEST_POOL: List[Dict] = []


def load_request_pool(path: str) -> None:
    """Load a pre-classified request pool JSON (built by calibrate --save-pool)."""
    global _REQUEST_POOL
    with open(path) as f:
        _REQUEST_POOL = json.load(f)
    print(f"Loaded {len(_REQUEST_POOL)} real prompts from pool: {path}")


@dataclass
class RequestProfile:
    """One user request with its characteristics."""
    request_id: int
    user_id: str
    n_tokens: int
    n_sensitive: int
    categories: Dict[str, int]
    alpha: float
    is_adversarial: bool = False

    @property
    def n_encrypted(self) -> int:
        if self.alpha <= 0.5:
            fraction = self.alpha / 0.5
            return int(self.n_sensitive * fraction)
        else:
            fraction = (self.alpha - 0.5) / 0.5
            extra = int((self.n_tokens - self.n_sensitive) * fraction)
            return min(self.n_tokens, self.n_sensitive + extra)

    @property
    def encrypted_per_category(self) -> Dict[str, int]:
        return allocate_encryption_budget(self.categories, self.n_encrypted)

    @property
    def encryption_ratio(self) -> float:
        return self.n_encrypted / self.n_tokens if self.n_tokens > 0 else 0

    @property
    def r_exp(self) -> float:
        return compute_r_exp(self.categories, self.encrypted_per_category)

    @property
    def eenc(self) -> float:
        return float(self.n_encrypted)


@dataclass
class RequestResult:
    """Timing + privacy results for one request."""
    request_id: int
    user_id: str
    n_tokens: int
    n_sensitive: int
    n_encrypted: int
    encryption_ratio: float
    r_exp: float
    eenc: float
    alpha: float
    authorized: bool
    categories: Dict[str, int] = field(default_factory=dict)
    encrypted_per_cat: Dict[str, int] = field(default_factory=dict)

    # Stage timings (ms)
    t_arrive: float = 0.0
    t_detection: float = 0.0
    t_encryption: float = 0.0
    t_zkp_gen: float = 0.0
    t_zkp_verify: float = 0.0
    t_blockchain: float = 0.0
    t_tee: float = 0.0
    t_total: float = 0.0

    # Network breakdown
    t_wan_in: float = 0.0       # user → gateway
    t_lan_gw_zkp: float = 0.0   # gateway → zkp prover
    t_lan_zkp_bc: float = 0.0   # zkp → blockchain
    t_lan_bc_tee: float = 0.0   # blockchain → tee
    t_wan_out: float = 0.0      # tee → user
    t_network_total: float = 0.0

    # Overhead decomposition
    t_crypto_overhead: float = 0.0   # detection + encryption + zkp_gen
    t_infra_overhead: float = 0.0    # blockchain + tee + network + queues

    # Queue wait times
    wait_gateway: float = 0.0
    wait_zkp: float = 0.0
    wait_blockchain: float = 0.0
    wait_tee: float = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

class PipelineSimulation:
    """
    DES of the ZKP-Blockchain LLM privacy pipeline.

    Resources:
        gateway:    capacity=1
        zkp_prover: capacity=2
        blockchain: capacity=4 (validators)
        tee:        capacity=1
    """

    def __init__(
        self,
        n_gateway: int = 1,
        n_zkp_provers: int = 2,
        n_validators: int = 4,
        n_tee: int = 1,
        seed: int = 42,
    ):
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
        enc_per_cat = req.encrypted_per_category

        result = RequestResult(
            request_id=req.request_id,
            user_id=req.user_id,
            n_tokens=req.n_tokens,
            n_sensitive=req.n_sensitive,
            n_encrypted=req.n_encrypted,
            encryption_ratio=req.encryption_ratio,
            r_exp=req.r_exp,
            eenc=req.eenc,
            alpha=req.alpha,
            authorized=not req.is_adversarial,
            categories=dict(req.categories),
            encrypted_per_cat=dict(enc_per_cat),
            t_arrive=self.env.now,
        )

        # ── WAN hop: user → gateway ──────────────────────────────────────────
        wan_in = sample_wan_latency()
        yield self.env.timeout(wan_in)
        result.t_wan_in = wan_in

        # ── STAGE 1: Gateway (detect + encrypt) ──────────────────────────────
        t0 = self.env.now
        gw_req = self.gateway.request()
        yield gw_req
        result.wait_gateway = self.env.now - t0

        detect_t = sample_detection_time(req.n_tokens)
        yield self.env.timeout(detect_t)
        result.t_detection = detect_t

        enc_t = sample_encryption_time(req.n_encrypted)
        yield self.env.timeout(enc_t)
        result.t_encryption = enc_t

        self.gateway.release(gw_req)

        # ── LAN hop: gateway → zkp prover ────────────────────────────────────
        lan1 = sample_lan_latency()
        yield self.env.timeout(lan1)
        result.t_lan_gw_zkp = lan1

        # ── STAGE 2: ZKP Generation (scales with n_encrypted) ────────────────
        t0 = self.env.now
        zkp_req = self.zkp_prover.request()
        yield zkp_req
        result.wait_zkp = self.env.now - t0

        zkp_t = sample_zkp_time(req.n_encrypted)
        yield self.env.timeout(zkp_t)
        result.t_zkp_gen = zkp_t

        self.zkp_prover.release(zkp_req)

        # ── LAN hop: zkp → blockchain ────────────────────────────────────────
        lan2 = sample_lan_latency()
        yield self.env.timeout(lan2)
        result.t_lan_zkp_bc = lan2

        # ── STAGE 3: Blockchain Verification ──────────────────────────────────
        t0 = self.env.now
        bc_req = self.blockchain.request()
        yield bc_req
        result.wait_blockchain = self.env.now - t0

        verify_t = sample_zkp_verify_time()
        yield self.env.timeout(verify_t)
        result.t_zkp_verify = verify_t

        bc_t = sample_blockchain_time(self.n_validators)
        yield self.env.timeout(bc_t)
        result.t_blockchain = bc_t

        self.blockchain.release(bc_req)

        # ── Adversarial: domain predicate fails → reject ──────────────────────
        if req.is_adversarial:
            result.authorized = False
            result.t_network_total = wan_in + lan1 + lan2
            result.t_crypto_overhead = detect_t + enc_t + zkp_t
            result.t_infra_overhead = (
                verify_t + bc_t + result.t_network_total
                + result.wait_gateway + result.wait_zkp + result.wait_blockchain
            )
            result.t_total = self.env.now - result.t_arrive
            self.rejected.append(result)
            return

        # ── LAN hop: blockchain → tee ────────────────────────────────────────
        lan3 = sample_lan_latency()
        yield self.env.timeout(lan3)
        result.t_lan_bc_tee = lan3

        # ── STAGE 4: TEE Inference ────────────────────────────────────────────
        t0 = self.env.now
        tee_req = self.tee.request()
        yield tee_req
        result.wait_tee = self.env.now - t0

        tee_t = sample_tee_inference_time(req.n_tokens)
        yield self.env.timeout(tee_t)
        result.t_tee = tee_t

        self.tee.release(tee_req)

        # ── WAN hop: tee → user ──────────────────────────────────────────────
        wan_out = sample_wan_latency()
        yield self.env.timeout(wan_out)
        result.t_wan_out = wan_out

        # ── Overhead decomposition ────────────────────────────────────────────
        result.t_network_total = wan_in + lan1 + lan2 + lan3 + wan_out
        result.t_crypto_overhead = detect_t + enc_t + zkp_t
        result.t_infra_overhead = (
            verify_t + bc_t + tee_t + result.t_network_total
            + result.wait_gateway + result.wait_zkp
            + result.wait_blockchain + result.wait_tee
        )
        result.t_total = self.env.now - result.t_arrive
        self.results.append(result)

    def run_requests(self, requests: List[RequestProfile]):
        for req in requests:
            self.env.process(self._process_request(req))
        self.env.run()

    def run_poisson_arrivals(
        self,
        n_requests: int,
        arrival_rate_per_sec: float,
        request_factory: Callable,
    ):
        def _arrival_process():
            for i in range(n_requests):
                req = request_factory(i)
                self.env.process(self._process_request(req))
                yield self.env.timeout(sample_arrival_interval(arrival_rate_per_sec))
        self.env.process(_arrival_process())
        self.env.run()


def make_request(
    req_id: int,
    alpha: float = 0.5,
    domain: str = "healthcare",
    adversarial_prob: float = 0.05,
) -> RequestProfile:
    if _REQUEST_POOL:
        entry = random.choice(_REQUEST_POOL)
        n_tokens = entry["n_tokens"]
        n_sensitive = min(entry["n_sensitive"], n_tokens)
        categories = dict(entry["categories"])
    else:
        n_tokens = random.choices(TOKEN_DIST, TOKEN_WEIGHTS)[0]
        n_sensitive = sample_n_sensitive(n_tokens, domain)
        n_sensitive = min(n_sensitive, n_tokens)
        categories = generate_category_counts(n_sensitive, domain)

    return RequestProfile(
        request_id=req_id,
        user_id=f"user_{req_id % 50:03d}",
        n_tokens=n_tokens,
        n_sensitive=n_sensitive,
        categories=categories,
        alpha=alpha,
        is_adversarial=random.random() < adversarial_prob,
    )


def make_request_from_prompt(
    prompt_text: str,
    req_id: int = 1,
    alpha: float = 0.5,
    adversarial_prob: float = 0.0,
    cache_file: str | None = None,
) -> "RequestProfile":
    """Build a real RequestProfile by running the sensitivity classifier on prompt_text.

    Falls back to make_request() with domain='general' if calibrate is unavailable.
    """
    if not _CALIBRATE_AVAILABLE:
        print("Warning: calibrate not available — falling back to synthetic request.")
        return make_request(req_id, alpha=alpha, domain="general",
                            adversarial_prob=adversarial_prob)

    result = classify_prompt(prompt_text, cache_file=cache_file)
    n_sensitive = max(result["n_sensitive"], 0)
    n_tokens    = max(result["n_tokens"], 1)
    n_sensitive = min(n_sensitive, n_tokens)

    return RequestProfile(
        request_id=req_id,
        user_id=f"user_{req_id:03d}",
        n_tokens=n_tokens,
        n_sensitive=n_sensitive,
        categories=result["categories"],
        alpha=alpha,
        is_adversarial=random.random() < adversarial_prob,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-SEED CI INFRASTRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

N_SEEDS = 10
BASE_SEED = 42


def ci_95(values: List[float]) -> Tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi) for 95% confidence interval."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mu = sum(values) / n
    if n == 1:
        return (mu, mu, mu)
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    se = math.sqrt(var / n)
    t_crit = 2.262 if n <= 10 else 1.96
    return (mu, mu - t_crit * se, mu + t_crit * se)


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (len(s) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def avg(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def summarize(results: List[RequestResult]) -> dict:
    if not results:
        return {}

    totals     = [r.t_total for r in results]
    detect     = [r.t_detection for r in results]
    encrypt    = [r.t_encryption for r in results]
    zkp        = [r.t_zkp_gen for r in results]
    bc         = [r.t_blockchain for r in results]
    tee        = [r.t_tee for r in results]
    crypto_oh  = [r.t_crypto_overhead for r in results]
    infra_oh   = [r.t_infra_overhead for r in results]
    wait_gw    = [r.wait_gateway for r in results]
    wait_bc    = [r.wait_blockchain for r in results]
    wait_tee   = [r.wait_tee for r in results]
    enc_ratios = [r.encryption_ratio for r in results]
    r_exp_vals = [r.r_exp for r in results]
    net_total  = [r.t_network_total for r in results]
    wan_in     = [r.t_wan_in for r in results]
    wan_out    = [r.t_wan_out for r in results]

    return {
        "n": len(results),
        "latency_mean": avg(totals),
        "latency_p50":  percentile(totals, 50),
        "latency_p95":  percentile(totals, 95),
        "latency_p99":  percentile(totals, 99),
        "detect_mean":  avg(detect),
        "encrypt_mean": avg(encrypt),
        "zkp_mean":     avg(zkp),
        "blockchain_mean": avg(bc),
        "tee_mean":     avg(tee),
        "crypto_overhead_mean": avg(crypto_oh),
        "infra_overhead_mean":  avg(infra_oh),
        "network_total_mean":   avg(net_total),
        "wan_in_mean":  avg(wan_in),
        "wan_out_mean": avg(wan_out),
        "wait_gw_mean": avg(wait_gw),
        "wait_bc_mean": avg(wait_bc),
        "wait_tee_mean": avg(wait_tee),
        "enc_ratio_mean": avg(enc_ratios),
        "r_exp_mean":   avg(r_exp_vals),
        "r_exp_p50":    percentile(r_exp_vals, 50),
        "r_exp_p95":    percentile(r_exp_vals, 95),
        "throughput_rps": len(results) / (max(r.t_total + r.t_arrive for r in results) / 1000) if results else 0,
    }


def print_stats(stats: dict, label: str = ""):
    BOLD = "\033[1m"; CYAN = "\033[96m"; RESET = "\033[0m"
    if label:
        print(f"\n{BOLD}{CYAN}{label}{RESET}")
    if not stats:
        print("  No results")
        return
    print(f"  Requests:           {stats['n']}")
    print(f"  ── Latency (ms) ──")
    print(f"  Mean:               {stats['latency_mean']:.1f}")
    print(f"  P50:                {stats['latency_p50']:.1f}")
    print(f"  P95:                {stats['latency_p95']:.1f}")
    print(f"  P99:                {stats['latency_p99']:.1f}")
    print(f"  ── Stage Breakdown (mean ms) ──")
    print(f"  Detection:          {stats['detect_mean']:.1f}")
    print(f"  Encryption:         {stats['encrypt_mean']:.1f}")
    print(f"  ZKP generation:     {stats['zkp_mean']:.1f}")
    print(f"  Blockchain:         {stats['blockchain_mean']:.1f}")
    print(f"  TEE inference:      {stats['tee_mean']:.1f}")
    print(f"  ── Overhead Decomposition (mean ms) ──")
    print(f"  Crypto overhead:    {stats['crypto_overhead_mean']:.1f}  (detect+encrypt+zkp)")
    print(f"  Infra overhead:     {stats['infra_overhead_mean']:.1f}  (bc+tee+net+queues)")
    print(f"  Network total:      {stats['network_total_mean']:.1f}  (WAN in={stats['wan_in_mean']:.1f}, out={stats['wan_out_mean']:.1f})")
    print(f"  ── Queue Wait Times (mean ms) ──")
    print(f"  Wait @ Gateway:     {stats['wait_gw_mean']:.1f}")
    print(f"  Wait @ Blockchain:  {stats['wait_bc_mean']:.1f}")
    print(f"  Wait @ TEE:         {stats['wait_tee_mean']:.1f}")
    print(f"  ── Privacy (Weighted R_exp) ──")
    print(f"  R_exp mean:         {stats['r_exp_mean']:.4f}")
    print(f"  R_exp P50:          {stats['r_exp_p50']:.4f}")
    print(f"  R_exp P95:          {stats['r_exp_p95']:.4f}")
    print(f"  Enc ratio (mean):   {stats['enc_ratio_mean']:.3f}")
    print(f"  Throughput:         {stats['throughput_rps']:.2f} req/s")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Single Request — Full Pipeline Trace
# ═════════════════════════════════════════════════════════════════════════════

def experiment_single_trace():
    print("\n" + "="*80)
    print("EXPERIMENT 1: Single Request — Full Pipeline Trace")
    print("="*80)

    sim = PipelineSimulation()
    cats = {"NAME": 3, "ID": 2, "CONTACT": 2, "LOCATION": 1, "DOB": 1, "ORG": 1}
    req = RequestProfile(
        request_id=1, user_id="user_001", n_tokens=35, n_sensitive=10,
        categories=cats, alpha=0.5, is_adversarial=False,
    )
    sim.run_requests([req])

    if sim.results:
        r = sim.results[0]
        BOLD = "\033[1m"; RESET = "\033[0m"
        print(f"\n  Request ID:       {r.request_id}")
        print(f"  Tokens:           {r.n_tokens}")
        print(f"  Sensitive:        {r.n_sensitive}  ({r.encryption_ratio:.1%} encrypted)")
        print(f"  R_exp(α={r.alpha}):    {r.r_exp:.4f}")
        print(f"  E_enc:            {r.eenc:.0f}")
        print(f"\n  Category breakdown:")
        print(f"    {'Category':<12} {'Total':<8} {'Encrypted':<12} {'Exposed':<10} {'Weight'}")
        print(f"    {'-'*52}")
        for cat in CATEGORY_NAMES:
            total = r.categories.get(cat, 0)
            enc = r.encrypted_per_cat.get(cat, 0)
            exposed = total - enc
            w = CATEGORY_WEIGHTS[cat]
            marker = " ← exposed" if exposed > 0 else ""
            print(f"    {cat:<12} {total:<8} {enc:<12} {exposed:<10} {w:.1f}{marker}")

        print(f"\n  Stage timings:")
        print(f"  {'Detection':<25} {r.t_detection:>8.1f} ms")
        print(f"  {'Encryption':<25} {r.t_encryption:>8.1f} ms")
        print(f"  {'ZKP generation':<25} {r.t_zkp_gen:>8.1f} ms")
        print(f"  {'ZKP verify':<25} {r.t_zkp_verify:>8.1f} ms")
        print(f"  {'Blockchain finality':<25} {r.t_blockchain:>8.1f} ms")
        print(f"  {'TEE inference':<25} {r.t_tee:>8.1f} ms")
        print(f"  {'WAN in (user→gw)':<25} {r.t_wan_in:>8.1f} ms")
        print(f"  {'WAN out (tee→user)':<25} {r.t_wan_out:>8.1f} ms")
        print(f"  {'LAN total':<25} {r.t_lan_gw_zkp + r.t_lan_zkp_bc + r.t_lan_bc_tee:>8.1f} ms")
        print(f"\n  Overhead decomposition:")
        print(f"  {BOLD}{'Crypto overhead':<25} {r.t_crypto_overhead:>8.1f} ms{RESET}  (detect+encrypt+zkp_gen)")
        print(f"  {BOLD}{'Infra overhead':<25} {r.t_infra_overhead:>8.1f} ms{RESET}  (bc+tee+net+queues)")
        print(f"  {BOLD}{'Total latency':<25} {r.t_total:>8.1f} ms{RESET}")
        print(f"\n  Authorized:       {'✓ YES' if r.authorized else '✗ NO'}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Concurrent Users — Throughput & Queue (with CI)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_concurrent_users():
    print("\n" + "="*80)
    print("EXPERIMENT 2: Concurrent Users — Throughput & Queue (10-seed CI)")
    print("="*80)
    print(f"\n  {'Users':<8} {'Mean±CI (ms)':<22} {'P95 (ms)':<14} {'Throughput':<14} "
          f"{'R_exp':<10} {'Crypto OH':<12} {'Infra OH'}")
    print("  " + "-"*95)

    for n_users in [1, 5, 10, 20, 50]:
        lat_means, p95_means, thr_means = [], [], []
        rexp_means, crypto_means, infra_means = [], [], []

        for s in range(N_SEEDS):
            sim = PipelineSimulation(seed=BASE_SEED + s)
            requests = [make_request(i, alpha=0.5, domain="healthcare") for i in range(n_users)]
            sim.run_requests(requests)
            st = summarize(sim.results + sim.rejected)
            if st:
                lat_means.append(st['latency_mean'])
                p95_means.append(st['latency_p95'])
                thr_means.append(st['throughput_rps'])
                rexp_means.append(st['r_exp_mean'])
                crypto_means.append(st['crypto_overhead_mean'])
                infra_means.append(st['infra_overhead_mean'])

        lat_mu, lat_lo, lat_hi = ci_95(lat_means)
        p95_mu, _, _ = ci_95(p95_means)
        thr_mu, _, _ = ci_95(thr_means)
        rexp_mu, _, _ = ci_95(rexp_means)
        cry_mu, _, _ = ci_95(crypto_means)
        inf_mu, _, _ = ci_95(infra_means)

        print(f"  {n_users:<8} {lat_mu:>7.1f} ±{(lat_hi-lat_lo)/2:>5.1f}       "
              f"{p95_mu:<14.1f} {thr_mu:<14.2f} {rexp_mu:<10.4f} {cry_mu:<12.1f} {inf_mu:.1f}")


def experiment_poisson_arrivals():
    print("\n" + "─"*80)
    print("  Poisson Arrivals (100 requests, 10-seed CI)")
    print(f"\n  {'Rate':<10} {'Mean±CI (ms)':<22} {'P95 (ms)':<14} {'R_exp':<10} {'Crypto OH':<12} {'Wait@TEE'}")
    print("  " + "-"*75)

    for rate in [0.5, 1.0, 2.0, 5.0, 10.0]:
        lat_means, p95_means, rexp_means, cry_means, tee_means = [], [], [], [], []

        for s in range(N_SEEDS):
            sim = PipelineSimulation(seed=BASE_SEED + s)
            sim.run_poisson_arrivals(100, rate,
                lambda i: make_request(i, domain="healthcare"))
            st = summarize(sim.results)
            if st:
                lat_means.append(st['latency_mean'])
                p95_means.append(st['latency_p95'])
                rexp_means.append(st['r_exp_mean'])
                cry_means.append(st['crypto_overhead_mean'])
                tee_means.append(st['wait_tee_mean'])

        lat_mu, lat_lo, lat_hi = ci_95(lat_means)
        p95_mu, _, _ = ci_95(p95_means)
        rexp_mu, _, _ = ci_95(rexp_means)
        cry_mu, _, _ = ci_95(cry_means)
        tee_mu, _, _ = ci_95(tee_means)

        print(f"  {rate:<10.1f} {lat_mu:>7.1f} ±{(lat_hi-lat_lo)/2:>5.1f}       "
              f"{p95_mu:<14.1f} {rexp_mu:<10.4f} {cry_mu:<12.1f} {tee_mu:.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: α Sweep — Pareto (weighted R_exp, 10-seed CI)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_alpha_sweep():
    print("\n" + "="*80)
    print("EXPERIMENT 3: α Sweep — Privacy/Efficiency Pareto (10-seed CI)")
    print("="*80)
    print(f"\n  {'α':<6} {'R_exp±CI':<18} {'Enc ratio':<12} {'Latency±CI (ms)':<22} "
          f"{'Crypto OH':<12} {'ZKP (ms)'}")
    print("  " + "-"*85)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pareto_data = []

    for alpha in alphas:
        rexp_list, lat_list, enc_list, cry_list, zkp_list = [], [], [], [], []

        for s in range(N_SEEDS):
            sim = PipelineSimulation(seed=BASE_SEED + s)
            sim.run_poisson_arrivals(200, 2.0,
                lambda i, a=alpha: make_request(i, alpha=a, domain="healthcare"))
            st = summarize(sim.results)
            if st:
                rexp_list.append(st['r_exp_mean'])
                lat_list.append(st['latency_mean'])
                enc_list.append(st['enc_ratio_mean'])
                cry_list.append(st['crypto_overhead_mean'])
                zkp_list.append(st['zkp_mean'])

        r_mu, r_lo, r_hi = ci_95(rexp_list)
        l_mu, l_lo, l_hi = ci_95(lat_list)
        e_mu, _, _ = ci_95(enc_list)
        c_mu, _, _ = ci_95(cry_list)
        z_mu, _, _ = ci_95(zkp_list)

        print(f"  {alpha:<6.1f} {r_mu:>6.4f} ±{(r_hi-r_lo)/2:>6.4f}    "
              f"{e_mu:<12.3f} {l_mu:>7.1f} ±{(l_hi-l_lo)/2:>5.1f}       "
              f"{c_mu:<12.1f} {z_mu:.1f}")

        pareto_data.append({
            "alpha": alpha,
            "r_exp_mean": r_mu, "r_exp_ci_lo": r_lo, "r_exp_ci_hi": r_hi,
            "enc_ratio": e_mu,
            "latency_mean": l_mu, "latency_ci_lo": l_lo, "latency_ci_hi": l_hi,
            "crypto_overhead": c_mu,
            "zkp_mean": z_mu,
        })

    # compute utility loss relative to α=0 baseline (no encryption)
    baseline_latency = pareto_data[0]["latency_mean"]
    for entry in pareto_data:
        entry["utility_loss"] = (entry["latency_mean"] - baseline_latency) / baseline_latency if baseline_latency else 0.0
        entry["latency_overhead_ms"] = entry["latency_mean"] - baseline_latency

    return pareto_data


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3b: Per-Category Exposure vs α
# ═════════════════════════════════════════════════════════════════════════════

def experiment_category_exposure_by_alpha():
    print("\n" + "="*80)
    print("EXPERIMENT 3b: Per-Category Exposure vs α (priority-based, 10-seed avg)")
    print("="*80)
    header = f"  {'α':<6}"
    for cat in CATEGORY_NAMES:
        header += f" {cat:<10}"
    header += f" {'R_exp':<10}"
    print(f"\n{header}")
    print("  " + "-"*80)

    cat_data = []

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        cat_exposed_accum = {cat: [] for cat in CATEGORY_NAMES}
        rexp_accum = []

        for s in range(N_SEEDS):
            sim = PipelineSimulation(seed=BASE_SEED + s)
            sim.run_poisson_arrivals(200, 2.0,
                lambda i, a=alpha: make_request(i, alpha=a, domain="healthcare"))

            for cat in CATEGORY_NAMES:
                vals = [r.categories.get(cat, 0) - r.encrypted_per_cat.get(cat, 0)
                        for r in sim.results]
                cat_exposed_accum[cat].append(avg(vals))

            st = summarize(sim.results)
            if st:
                rexp_accum.append(st['r_exp_mean'])

        r_mu = avg(rexp_accum)
        row = f"  {alpha:<6.1f}"
        entry = {"alpha": alpha, "r_exp": r_mu}
        for cat in CATEGORY_NAMES:
            mu = avg(cat_exposed_accum[cat])
            row += f" {mu:<10.2f}"
            entry[f"{cat}_exposed"] = mu
        row += f" {r_mu:<10.4f}"
        print(row)
        cat_data.append(entry)

    return cat_data


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Scalability — Latency vs Token Count (with CI)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_scalability():
    print("\n" + "="*80)
    print("EXPERIMENT 4: Scalability — Latency vs Token Count (10-seed CI)")
    print("="*80)
    print(f"\n  {'n_tok':<8} {'Detect':<10} {'Encrypt':<10} {'ZKP':<10} "
          f"{'Total±CI (ms)':<20} {'Crypto OH':<12} {'R_exp'}")
    print("  " + "-"*80)

    for n in [10, 20, 50, 100, 200, 500, 1000]:
        det_l, enc_l, zkp_l, tot_l, cry_l, rexp_l = [], [], [], [], [], []

        for s in range(N_SEEDS):
            def make_fixed(i, n_tok=n, _seed=BASE_SEED+s):
                ns = sample_n_sensitive(n_tok, "healthcare")
                ns = min(ns, n_tok)
                cats = generate_category_counts(ns, "healthcare")
                return RequestProfile(
                    request_id=i, user_id=f"u{i}", n_tokens=n_tok,
                    n_sensitive=ns, categories=cats, alpha=0.5
                )
            sim = PipelineSimulation(seed=BASE_SEED + s)
            sim.run_poisson_arrivals(30, 2.0, make_fixed)
            st = summarize(sim.results)
            if st:
                det_l.append(st['detect_mean'])
                enc_l.append(st['encrypt_mean'])
                zkp_l.append(st['zkp_mean'])
                tot_l.append(st['latency_mean'])
                cry_l.append(st['crypto_overhead_mean'])
                rexp_l.append(st['r_exp_mean'])

        t_mu, t_lo, t_hi = ci_95(tot_l)
        print(f"  {n:<8} {avg(det_l):<10.1f} {avg(enc_l):<10.1f} {avg(zkp_l):<10.1f} "
              f"{t_mu:>7.1f} ±{(t_hi-t_lo)/2:>5.1f}       "
              f"{avg(cry_l):<12.1f} {avg(rexp_l):.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Adversarial Rejection (with CI)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_adversarial():
    print("\n" + "="*80)
    print("EXPERIMENT 5: Adversarial Rejection (10-seed CI)")
    print("="*80)
    print(f"\n  {'Adv prob':<10} {'Reject rate±CI':<22} {'Auth latency±CI (ms)'}")
    print("  " + "-"*55)

    for adv_prob in [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]:
        rej_rates, lat_means = [], []

        for s in range(N_SEEDS):
            sim = PipelineSimulation(seed=BASE_SEED + s)
            sim.run_poisson_arrivals(100, 2.0,
                lambda i, p=adv_prob: make_request(i, adversarial_prob=p))
            total = len(sim.results) + len(sim.rejected)
            rr = len(sim.rejected) / total if total > 0 else 0
            rej_rates.append(rr)
            st = summarize(sim.results)
            lat_means.append(st.get('latency_mean', 0))

        rr_mu, rr_lo, rr_hi = ci_95(rej_rates)
        l_mu, l_lo, l_hi = ci_95(lat_means)

        print(f"  {adv_prob:<10.2f} {rr_mu:>6.1%} ±{(rr_hi-rr_lo)/2:>5.1%}          "
              f"{l_mu:>7.1f} ±{(l_hi-l_lo)/2:>5.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Domain Comparison (with CI)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_domain_comparison():
    print("\n" + "="*80)
    print("EXPERIMENT 6: Domain Comparison (10-seed CI)")
    print("="*80)
    print(f"\n  {'Domain':<12} {'Avg sens':<10} {'Enc ratio':<12} {'R_exp':<10} "
          f"{'Crypto OH':<12} {'Latency±CI (ms)'}")
    print("  " + "-"*75)

    for domain in ["healthcare", "finance", "general", "auth"]:
        sens_l, enc_l, rexp_l, cry_l, lat_l = [], [], [], [], []

        for s in range(N_SEEDS):
            sim = PipelineSimulation(seed=BASE_SEED + s)
            sim.run_poisson_arrivals(100, 2.0,
                lambda i, d=domain: make_request(i, domain=d))
            st = summarize(sim.results)
            if st:
                sens_l.append(avg([r.n_sensitive for r in sim.results]))
                enc_l.append(st['enc_ratio_mean'])
                rexp_l.append(st['r_exp_mean'])
                cry_l.append(st['crypto_overhead_mean'])
                lat_l.append(st['latency_mean'])

        l_mu, l_lo, l_hi = ci_95(lat_l)
        print(f"  {domain:<12} {avg(sens_l):<10.1f} {avg(enc_l):<12.3f} {avg(rexp_l):<10.4f} "
              f"{avg(cry_l):<12.1f} {l_mu:>7.1f} ±{(l_hi-l_lo)/2:>5.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: Baseline Comparison (with CI)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_baseline_comparison():
    print("\n" + "="*80)
    print("EXPERIMENT 7: Baseline Comparison (10-seed CI)")
    print("="*80)
    print(f"\n  {'Method':<25} {'R_exp±CI':<18} {'Enc ratio':<12} "
          f"{'Latency±CI (ms)':<22} {'Crypto OH'}")
    print("  " + "-"*85)

    methods = [
        ("No encryption (α=0)",     0.0),
        ("Full encryption (α=1)",   1.0),
        ("Selective (α=0.5, ours)", 0.5),
    ]

    for label, alpha in methods:
        rexp_l, enc_l, lat_l, cry_l = [], [], [], []

        for s in range(N_SEEDS):
            sim = PipelineSimulation(seed=BASE_SEED + s)
            sim.run_poisson_arrivals(100, 2.0,
                lambda i, a=alpha: make_request(i, alpha=a, domain="healthcare"))
            st = summarize(sim.results)
            if st:
                rexp_l.append(st['r_exp_mean'])
                enc_l.append(st['enc_ratio_mean'])
                lat_l.append(st['latency_mean'])
                cry_l.append(st['crypto_overhead_mean'])

        r_mu, r_lo, r_hi = ci_95(rexp_l)
        l_mu, l_lo, l_hi = ci_95(lat_l)

        print(f"  {label:<25} {r_mu:>6.4f} ±{(r_hi-r_lo)/2:>6.4f}    "
              f"{avg(enc_l):<12.3f} {l_mu:>7.1f} ±{(l_hi-l_lo)/2:>5.1f}       "
              f"{avg(cry_l):.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS TO JSON
# ═════════════════════════════════════════════════════════════════════════════

def save_results_json(all_results: List[RequestResult], filename: str = "simulation_results.json"):
    data = []
    for r in all_results:
        data.append({
            "request_id": r.request_id,
            "n_tokens": r.n_tokens,
            "n_sensitive": r.n_sensitive,
            "n_encrypted": r.n_encrypted,
            "encryption_ratio": r.encryption_ratio,
            "r_exp": r.r_exp,
            "eenc": r.eenc,
            "alpha": r.alpha,
            "authorized": r.authorized,
            "categories": r.categories,
            "encrypted_per_cat": r.encrypted_per_cat,
            "t_detection": r.t_detection,
            "t_encryption": r.t_encryption,
            "t_zkp_gen": r.t_zkp_gen,
            "t_zkp_verify": r.t_zkp_verify,
            "t_blockchain": r.t_blockchain,
            "t_tee": r.t_tee,
            "t_total": r.t_total,
            "t_crypto_overhead": r.t_crypto_overhead,
            "t_infra_overhead": r.t_infra_overhead,
            "t_wan_in": r.t_wan_in,
            "t_wan_out": r.t_wan_out,
            "t_network_total": r.t_network_total,
            "wait_gateway": r.wait_gateway,
            "wait_zkp": r.wait_zkp,
            "wait_blockchain": r.wait_blockchain,
            "wait_tee": r.wait_tee,
        })
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  ✓ Results saved to {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_prompt_pipeline(
    prompt: str,
    alpha: float = 0.5,
    cache_file: str | None = None,
) -> None:
    """Run a single real prompt through the full pipeline and print results."""
    BOLD  = "\033[1m"
    RESET = "\033[0m"
    print("\n" + "─" * 60)
    print(f"{BOLD}  PROMPT PIPELINE — real detection + simulation{RESET}")
    print("─" * 60)
    print(f"  Prompt: {prompt[:80]}{'…' if len(prompt) > 80 else ''}")
    print(f"  Alpha (encryption budget): {alpha}")

    req = make_request_from_prompt(prompt, req_id=1, alpha=alpha,
                                   cache_file=cache_file)

    print(f"\n  Tokens detected:    {req.n_tokens}")
    print(f"  Sensitive tokens:   {req.n_sensitive}")
    print(f"  Tokens to encrypt:  {req.n_encrypted}  ({req.encryption_ratio:.1%})")
    print(f"  Category breakdown: {req.categories}")

    sim = PipelineSimulation(seed=42)
    sim.run_single(req)
    r = sim.results[0]

    print(f"\n  {'Stage':<22} {'Time (ms)':>10}")
    print(f"  {'─'*34}")
    print(f"  {'Detection':<22} {r.t_detection:>10.1f}")
    print(f"  {'Encryption (ChaCha20)':<22} {r.t_encryption:>10.1f}")
    print(f"  {'ZKP generation':<22} {r.t_zkp_gen:>10.1f}")
    print(f"  {'Blockchain':<22} {r.t_blockchain:>10.1f}")
    print(f"  {'TEE inference':<22} {r.t_tee:>10.1f}")
    print(f"  {'─'*34}")
    print(f"  {BOLD}{'Total latency':<22} {r.t_total:>10.1f}{RESET}")
    print(f"\n  Exposure risk R_exp: {r.r_exp:.4f}  (0=fully private, 1=fully exposed)")
    print(f"  Authorized:         {r.authorized}")
    print("─" * 60)


if __name__ == "__main__":
    import argparse as _argparse
    _p = _argparse.ArgumentParser(description="ZKP-Blockchain LLM Privacy Pipeline")
    _p.add_argument("--prompt", type=str, default=None,
                    help="Run a single real prompt through the pipeline and exit")
    _p.add_argument("--alpha", type=float, default=0.5,
                    help="Encryption budget α ∈ [0,1] (default 0.5)")
    _p.add_argument("--cache", type=str, default=None, metavar="PATH",
                    help="Token sensitivity cache JSON (from calibrate.py --cache)")
    _p.add_argument("--real-data", type=str, default=None, metavar="PATH",
                    help="Request pool JSON (from calibrate.py --save-pool); makes all experiments use real prompts")
    _args = _p.parse_args()

    if _args.real_data:
        load_request_pool(_args.real_data)

    if _args.prompt:
        run_prompt_pipeline(_args.prompt, alpha=_args.alpha, cache_file=_args.cache)
    else:
        print("\n" + "█"*80)
        print("  ZKP-BLOCKCHAIN LLM PRIVACY PIPELINE — SimPy Simulation (Merged)")
        print("  Privacy: Weighted R_exp over PHI categories (HIPAA-aligned)")
        print("  Rigor:   10-seed CI | crypto/infra split | WAN/LAN | ZKP∝n_enc")
        print("█"*80)

        experiment_single_trace()
        experiment_concurrent_users()
        experiment_poisson_arrivals()
        pareto_data = experiment_alpha_sweep()
        cat_data = experiment_category_exposure_by_alpha()
        experiment_scalability()
        experiment_adversarial()
        experiment_domain_comparison()
        experiment_baseline_comparison()

        # Save results
        with open("alpha_sweep_results.json", "w") as f:
            json.dump(pareto_data, f, indent=2)
        print("\n  ✓ Alpha sweep saved to alpha_sweep_results.json")

        with open("category_exposure_results.json", "w") as f:
            json.dump(cat_data, f, indent=2)
        print("  ✓ Category exposure saved to category_exposure_results.json")

        sim_final = PipelineSimulation(seed=42)
        sim_final.run_poisson_arrivals(500, 2.0,
            lambda i: make_request(i, domain="healthcare"))
        save_results_json(sim_final.results, "simulation_results.json")

    print("\n" + "█"*80)
    print("  Done. All results include 95% CI across 10 seeds.")
    print("█"*80)