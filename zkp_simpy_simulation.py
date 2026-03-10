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

Usage:
    # With real SimPy installed:
    pip install simpy
    python3 zkp_simpy_simulation.py

    # Works identically without SimPy (built-in DES engine):
    python3 zkp_simpy_simulation.py

Scenarios run:
    1. Single request — full pipeline trace
    2. Concurrent users — throughput + queue analysis
    3. α sweep — Privacy/efficiency Pareto
    4. Scalability — latency vs token count
    5. Adversarial — rejection rate for malicious prompts
"""

import random
import math
import json
import time
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Generator

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
            self._heap = []   # (time, counter, callback, event)
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
# REALISTIC TIMING DISTRIBUTIONS
# Sourced from: RFC 7539, zkSNARK benchmarks, Hyperledger Fabric docs,
# Intel SGX whitepapers, spaCy benchmarks
# ═════════════════════════════════════════════════════════════════════════════

def sample_detection_time(n_tokens: int) -> float:
    """
    Token sensitivity classification (Algorithm 1).
    Base: ~2ms + 0.08ms per token (spaCy ~10k tokens/sec for en_core_web_sm).
    Cache hit rate ~60% reduces NLP calls significantly.
    Returns time in ms.
    """
    base = random.gauss(2.0, 0.5)
    per_token = n_tokens * random.gauss(0.08, 0.01)
    return max(0.5, base + per_token)


def sample_encryption_time(n_encrypted: int) -> float:
    """
    ChaCha20-Poly1305 selective encryption.
    ~0.1ms per token (conservative; ChaCha20 does ~1GB/s on modern hardware,
    typical token ~4 bytes → ~0.004μs, but Python overhead dominates).
    Returns time in ms.
    """
    base = random.gauss(0.5, 0.1)
    per_token = n_encrypted * random.gauss(0.1, 0.02)
    return max(0.1, base + per_token)


def sample_zkp_time(n_constraints: int = 500) -> float:
    """
    ZKP generation at gateway (Groth16/PLONK).
    Literature: Groth16 prove time ~100–300ms for circuits with ~10^3–10^5 constraints.
    Our circuit: 5 predicates, ~500 constraints estimated.
    Ref: zkSNARK benchmarks [11,12] in the paper.
    Returns time in ms.
    """
    # Log-normal to model realistic variance in proof generation
    mu = math.log(150)   # median 150ms
    sigma = 0.3
    return random.lognormvariate(mu, sigma)


def sample_zkp_verify_time() -> float:
    """
    ZKP verification on blockchain (Groth16 verify is O(1) ~2–5ms).
    Much faster than proving.
    Returns time in ms.
    """
    return random.gauss(3.0, 0.5)


def sample_blockchain_time(n_validators: int = 4) -> float:
    """
    Hyperledger Fabric permissioned blockchain finality.
    Typical: 500–2000ms depending on validator count + consensus round.
    With 4 validators (BFT threshold f < 1): ~800ms median.
    Ref: Hyperledger Fabric performance benchmarks.
    Returns time in ms.
    """
    base = random.gauss(800, 150)
    validator_overhead = n_validators * random.gauss(20, 5)
    return max(200, base + validator_overhead)


def sample_tee_inference_time(n_tokens: int) -> float:
    """
    TEE-based LLM inference (Intel SGX) on cloud VM (Azure DCsv3).
    Base latency reduced vs local: cloud SGX ~5ms base + 0.3ms/token.
    SGX overhead ~12% over native remains.
    Returns time in ms.
    """
    base = random.gauss(5, 1.0)
    per_token = n_tokens * random.gauss(0.3, 0.05)
    sgx_overhead = 1.12   # 12% SGX overhead
    return max(1.0, (base + per_token) * sgx_overhead)


def sample_network_latency() -> float:
    """
    Network round-trip between components (LAN assumed, same datacenter).
    ~0.5–2ms per hop.
    Returns time in ms.
    """
    return max(0.1, random.gauss(1.0, 0.3))


def sample_arrival_interval(rate_per_sec: float) -> float:
    """
    Poisson arrival process. Returns inter-arrival time in ms.
    """
    return random.expovariate(rate_per_sec) * 1000


# ═════════════════════════════════════════════════════════════════════════════
# REQUEST DATA MODEL
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RequestProfile:
    """Represents one user request with its characteristics."""
    request_id: int
    user_id: str
    n_tokens: int
    n_sensitive: int          # |S(P)| — sensitive token count
    alpha: float              # privacy-efficiency trade-off
    is_adversarial: bool = False  # triggers domain predicate failure

    @property
    def n_encrypted(self) -> int:
        """
        α controls privacy/efficiency tradeoff:
          α=0.0 → encrypt nothing (pure efficiency)
          α=0.5 → encrypt all sensitive tokens (baseline)
          α=1.0 → encrypt all tokens (pure privacy)
        Linear interpolation between these anchors.
        """
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
        """Exposed sensitive tokens (not encrypted)."""
        return float(max(0, self.n_sensitive - self.n_encrypted))

    @property
    def eenc(self) -> float:
        """Encrypted tokens (may include non-sensitive when α>0.5)."""
        return float(self.n_encrypted)


@dataclass
class RequestResult:
    """Timing results for one request — maps to Table III."""
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

    # Stage timings (ms)
    t_arrive: float = 0.0
    t_detection: float = 0.0
    t_encryption: float = 0.0
    t_zkp_gen: float = 0.0
    t_blockchain: float = 0.0
    t_tee: float = 0.0
    t_network_total: float = 0.0
    t_total: float = 0.0

    # Queue wait times
    wait_gateway: float = 0.0
    wait_zkp: float = 0.0
    wait_blockchain: float = 0.0
    wait_tee: float = 0.0

    zkp_proof_size_bytes: int = 32


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

class PipelineSimulation:
    """
    Discrete-event simulation of the ZKP-Blockchain LLM privacy pipeline.

    Resources (SimPy):
        gateway:    capacity=1 (single trusted gateway)
        zkp_prover: capacity=2 (can prove 2 requests in parallel)
        blockchain: capacity=4 (4 validator nodes, BFT)
        tee:        capacity=1 (single TEE enclave, GPU-bound)
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

        # SimPy resources — each models a real system component
        self.gateway    = simpy.Resource(self.env, capacity=n_gateway)
        self.zkp_prover = simpy.Resource(self.env, capacity=n_zkp_provers)
        self.blockchain = simpy.Resource(self.env, capacity=n_validators)
        self.tee        = simpy.Resource(self.env, capacity=n_tee)

        self.n_validators = n_validators
        self.results: List[RequestResult] = []
        self.rejected: List[RequestResult] = []

    def _process_request(self, req: RequestProfile) -> Generator:
        """
        SimPy process: models one request flowing through all pipeline stages.
        Uses explicit request/release (compatible with both real SimPy and DES engine).
        """
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
            t_arrive=self.env.now,
        )

        # ── STAGE 1: Gateway (detect + encrypt) ──────────────────────────────
        t_before_wait = self.env.now
        gw_req = self.gateway.request()
        yield gw_req
        result.wait_gateway = self.env.now - t_before_wait

        net1 = sample_network_latency()
        yield self.env.timeout(net1)

        detect_t = sample_detection_time(req.n_tokens)
        yield self.env.timeout(detect_t)
        result.t_detection = detect_t

        enc_t = sample_encryption_time(req.n_encrypted)
        yield self.env.timeout(enc_t)
        result.t_encryption = enc_t

        self.gateway.release(gw_req)

        # ── STAGE 2: ZKP Generation ───────────────────────────────────────────
        t_before_wait = self.env.now
        zkp_req = self.zkp_prover.request()
        yield zkp_req
        result.wait_zkp = self.env.now - t_before_wait

        zkp_t = sample_zkp_time()
        yield self.env.timeout(zkp_t)
        result.t_zkp_gen = zkp_t

        self.zkp_prover.release(zkp_req)

        # ── STAGE 3: Blockchain Verification (eq. 9–11) ───────────────────────
        t_before_wait = self.env.now
        bc_req = self.blockchain.request()
        yield bc_req
        result.wait_blockchain = self.env.now - t_before_wait

        net2 = sample_network_latency()
        yield self.env.timeout(net2)

        verify_t = sample_zkp_verify_time()
        yield self.env.timeout(verify_t)

        bc_t = sample_blockchain_time(self.n_validators)
        yield self.env.timeout(bc_t)
        result.t_blockchain = verify_t + bc_t

        self.blockchain.release(bc_req)

        # ── Adversarial check: domain predicate fails → reject ────────────────
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

        net3 = sample_network_latency()
        yield self.env.timeout(net3)

        tee_t = sample_tee_inference_time(req.n_tokens)
        yield self.env.timeout(tee_t)
        result.t_tee = tee_t

        self.tee.release(tee_req)

        # ── STAGE 5: Response back to user ────────────────────────────────────
        net4 = sample_network_latency()
        yield self.env.timeout(net4)

        result.t_network_total = net1 + net2 + net3 + net4
        result.t_total = self.env.now - result.t_arrive
        self.results.append(result)

    def run_requests(self, requests: List[RequestProfile]):
        """Schedule all requests and run the simulation."""
        for req in requests:
            self.env.process(self._process_request(req))
        self.env.run()

    def run_poisson_arrivals(
        self,
        n_requests: int,
        arrival_rate_per_sec: float,
        request_factory,
    ):
        """
        Generate requests with Poisson inter-arrival times.
        arrival_rate_per_sec: e.g., 5.0 = 5 requests per second
        """
        def _arrival_process():
            for i in range(n_requests):
                req = request_factory(i)
                self.env.process(self._process_request(req))
                inter_arrival = sample_arrival_interval(arrival_rate_per_sec)
                yield self.env.timeout(inter_arrival)

        self.env.process(_arrival_process())
        self.env.run()


TOKEN_DIST    = [50, 100, 150, 175, 200, 225, 250, 300, 400, 500]
TOKEN_WEIGHTS = [5,   10,  15,  20,  20,  15,   8,   4,   2,   1]


SENSITIVE_RATIOS = {
    "healthcare": (0.06, 0.09),   
    "finance":    (0.15, 0.30),   
    "general":    (0.05, 0.15),   
    "auth":       (0.30, 0.50),   
}


def make_request(
    req_id: int,
    alpha: float = 0.5,
    domain: str = "healthcare",
    adversarial_prob: float = 0.05,
) -> RequestProfile:
    n_tokens = random.choices(TOKEN_DIST, TOKEN_WEIGHTS)[0]
    lo, hi = SENSITIVE_RATIOS[domain]
    sens_ratio = random.uniform(lo, hi)
    n_sensitive = max(1, int(n_tokens * sens_ratio))
    is_adv = random.random() < adversarial_prob

    return RequestProfile(
        request_id=req_id,
        user_id=f"user_{req_id % 50:03d}",
        n_tokens=n_tokens,
        n_sensitive=n_sensitive,
        alpha=alpha,
        is_adversarial=is_adv,
    )


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


def summarize(results: List[RequestResult], label: str = ""):
    if not results:
        print(f"  {label}: No results")
        return {}

    totals     = [r.t_total for r in results]
    detect     = [r.t_detection for r in results]
    encrypt    = [r.t_encryption for r in results]
    zkp        = [r.t_zkp_gen for r in results]
    bc         = [r.t_blockchain for r in results]
    tee        = [r.t_tee for r in results]
    wait_gw    = [r.wait_gateway for r in results]
    wait_bc    = [r.wait_blockchain for r in results]
    wait_tee   = [r.wait_tee for r in results]
    enc_ratios = [r.encryption_ratio for r in results]

    def avg(lst): return sum(lst) / len(lst) if lst else 0

    stats = {
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
        "wait_gw_mean": avg(wait_gw),
        "wait_bc_mean": avg(wait_bc),
        "wait_tee_mean":avg(wait_tee),
        "enc_ratio_mean": avg(enc_ratios),
        "throughput_rps": len(results) / (max(r.t_total + r.t_arrive for r in results) / 1000) if results else 0,
    }
    return stats


def print_stats(stats: dict, label: str = ""):
    BOLD = "\033[1m"; CYAN = "\033[96m"; GREEN = "\033[92m"; RESET = "\033[0m"
    if label:
        print(f"\n{BOLD}{CYAN}{label}{RESET}")
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
    print(f"  ── Queue Wait Times (mean ms) ──")
    print(f"  Wait @ Gateway:     {stats['wait_gw_mean']:.1f}")
    print(f"  Wait @ Blockchain:  {stats['wait_bc_mean']:.1f}")
    print(f"  Wait @ TEE:         {stats['wait_tee_mean']:.1f}")
    print(f"  ── Privacy ──")
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
    req = RequestProfile(
        request_id=1,
        user_id="user_001",
        n_tokens=35,
        n_sensitive=10,
        alpha=0.5,
        is_adversarial=False,
    )
    sim.run_requests([req])

    if sim.results:
        r = sim.results[0]
        BOLD = "\033[1m"; RESET = "\033[0m"
        print(f"\n  Request ID:       {r.request_id}")
        print(f"  Tokens:           {r.n_tokens}")
        print(f"  Sensitive:        {r.n_sensitive}  ({r.encryption_ratio:.1%})")
        print(f"  E_exp:            {r.eexp:.0f}  (eq. 26)")
        print(f"  E_enc:            {r.eenc:.0f}  (eq. 27)")
        print(f"\n  Stage timings:")
        print(f"  {'Detection':<25} {r.t_detection:>8.1f} ms")
        print(f"  {'Encryption':<25} {r.t_encryption:>8.1f} ms")
        print(f"  {'ZKP generation':<25} {r.t_zkp_gen:>8.1f} ms  ← dominates")
        print(f"  {'Blockchain+verify':<25} {r.t_blockchain:>8.1f} ms  ← dominates")
        print(f"  {'TEE inference':<25} {r.t_tee:>8.1f} ms")
        print(f"  {'Network (total)':<25} {r.t_network_total:>8.1f} ms")
        print(f"  {BOLD}{'Total latency':<25} {r.t_total:>8.1f} ms{RESET}")
        print(f"\n  Authorized:       {'✓ YES' if r.authorized else '✗ NO'}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Concurrent Users — Throughput & Queue Analysis
# ═════════════════════════════════════════════════════════════════════════════

def experiment_concurrent_users():
    print("\n" + "="*80)
    print("EXPERIMENT 2: Concurrent Users — Throughput & Queue Analysis")
    print("="*80)
    print(f"\n  {'Users/rate':<15} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput':<14} {'Wait@BC (ms)'}")
    print("  " + "-"*65)

    for n_users in [1, 5, 10, 20, 50]:
        sim = PipelineSimulation(seed=42)
        requests = [make_request(i, alpha=0.5, domain="healthcare") for i in range(n_users)]
        sim.run_requests(requests)
        stats = summarize(sim.results + sim.rejected)
        if stats:
            print(f"  {n_users:<15} {stats['latency_mean']:<12.1f} {stats['latency_p95']:<12.1f} "
                  f"{stats['throughput_rps']:<14.2f} {stats['wait_bc_mean']:.1f}")


def experiment_poisson_arrivals():
    print("\n" + "─"*80)
    print("  Poisson Arrivals (100 requests, varying arrival rate)")
    print(f"\n  {'Rate (req/s)':<15} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Queue@TEE (ms)'}")
    print("  " + "-"*55)

    for rate in [0.5, 1.0, 2.0, 5.0, 10.0]:
        sim = PipelineSimulation(seed=42)
        sim.run_poisson_arrivals(
            n_requests=100,
            arrival_rate_per_sec=rate,
            request_factory=lambda i: make_request(i, domain="healthcare"),
        )
        stats = summarize(sim.results)
        if stats:
            print(f"  {rate:<15.1f} {stats['latency_mean']:<12.1f} {stats['latency_p95']:<12.1f} "
                  f"{stats['wait_tee_mean']:.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: α Sweep — Privacy/Efficiency Pareto (Section V.B.2)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_alpha_sweep():
    print("\n" + "="*80)
    print("EXPERIMENT 3: α Sweep — Privacy/Efficiency Pareto (eq. 28)")
    print("="*80)
    print(f"\n  {'α':<8} {'E_exp':<10} {'E_enc':<10} {'Enc ratio':<12} {'Latency (ms)':<14} {'P95 (ms)'}")
    print("  " + "-"*65)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pareto_data = []

    for alpha in alphas:
        sim = PipelineSimulation(seed=42)
        sim.run_poisson_arrivals(
            n_requests=200,
            arrival_rate_per_sec=2.0,
            request_factory=lambda i, a=alpha: make_request(i, alpha=a, domain="healthcare"),
        )
        stats = summarize(sim.results)

        avg_eexp = sum(r.eexp for r in sim.results) / max(len(sim.results), 1)
        avg_eenc = sum(r.eenc for r in sim.results) / max(len(sim.results), 1)

        if stats:
            print(f"  {alpha:<8.1f} {avg_eexp:<10.2f} {avg_eenc:<10.2f} "
                  f"{stats['enc_ratio_mean']:<12.3f} {stats['latency_mean']:<14.1f} "
                  f"{stats['latency_p95']:.1f}")
            pareto_data.append({
                "alpha": alpha,
                "eexp": avg_eexp,
                "eenc": avg_eenc,
                "enc_ratio": stats['enc_ratio_mean'],
                "latency_mean": stats['latency_mean'],
                "latency_p95": stats['latency_p95'],
            })

    return pareto_data


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Scalability — Latency vs. Token Count (Section V.B.4)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_scalability():
    print("\n" + "="*80)
    print("EXPERIMENT 4: Scalability — Latency vs. Token Count n")
    print("="*80)
    print(f"\n  {'n_tokens':<12} {'Detect (ms)':<14} {'Encrypt (ms)':<14} {'ZKP (ms)':<12} {'Total (ms)':<12} {'Enc ratio'}")
    print("  " + "-"*75)

    token_counts = [10, 20, 50, 100, 200, 500, 1000]

    for n in token_counts:
        def make_fixed_token_req(i, n_tok=n):
            lo, hi = SENSITIVE_RATIOS["healthcare"]
            sens = max(1, int(n_tok * random.uniform(lo, hi)))
            return RequestProfile(
                request_id=i, user_id=f"u{i}", n_tokens=n_tok,
                n_sensitive=sens, alpha=0.5
            )
        sim = PipelineSimulation(seed=42)
        sim.run_poisson_arrivals(30, 2.0, make_fixed_token_req)
        stats = summarize(sim.results)
        if stats:
            print(f"  {n:<12} {stats['detect_mean']:<14.1f} {stats['encrypt_mean']:<14.1f} "
                  f"{stats['zkp_mean']:<12.1f} {stats['latency_mean']:<12.1f} "
                  f"{stats['enc_ratio_mean']:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Adversarial Rejection Rate (Section III.D — Adversary 3)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_adversarial():
    print("\n" + "="*80)
    print("EXPERIMENT 5: Adversarial Rejection — ZKP Domain Predicate (eq. 15)")
    print("="*80)
    print(f"\n  {'Adv prob':<12} {'Total':<8} {'Authorized':<12} {'Rejected':<10} {'Reject rate':<12} {'Latency (ms)'}")
    print("  " + "-"*65)

    for adv_prob in [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]:
        sim = PipelineSimulation(seed=42)
        sim.run_poisson_arrivals(
            n_requests=100,
            arrival_rate_per_sec=2.0,
            request_factory=lambda i, p=adv_prob: make_request(i, adversarial_prob=p),
        )
        n_auth = len(sim.results)
        n_rej  = len(sim.rejected)
        total  = n_auth + n_rej
        rej_rate = n_rej / total if total > 0 else 0
        stats = summarize(sim.results)
        lat = stats.get('latency_mean', 0) if stats else 0
        print(f"  {adv_prob:<12.2f} {total:<8} {n_auth:<12} {n_rej:<10} {rej_rate:<12.1%} {lat:.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Domain Comparison (healthcare vs finance vs general)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_domain_comparison():
    print("\n" + "="*80)
    print("EXPERIMENT 6: Domain Comparison — Encryption Load")
    print("="*80)
    print(f"\n  {'Domain':<14} {'Avg sensitive':<16} {'Enc ratio':<12} {'E_enc':<10} {'Latency (ms)'}")
    print("  " + "-"*60)

    for domain in ["healthcare", "finance", "general", "auth"]:
        sim = PipelineSimulation(seed=42)
        sim.run_poisson_arrivals(
            n_requests=100,
            arrival_rate_per_sec=2.0,
            request_factory=lambda i, d=domain: make_request(i, domain=d),
        )
        stats = summarize(sim.results)
        avg_sens = sum(r.n_sensitive for r in sim.results) / max(len(sim.results), 1)
        avg_eenc = sum(r.eenc for r in sim.results) / max(len(sim.results), 1)
        if stats:
            print(f"  {domain:<14} {avg_sens:<16.1f} {stats['enc_ratio_mean']:<12.3f} "
                  f"{avg_eenc:<10.1f} {stats['latency_mean']:.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: Baseline Comparison (no enc vs selective vs full enc)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_baseline_comparison():
    """
    Compares three baselines for paper Section V:
      (1) No encryption
      (2) Full encryption (all tokens)
      (3) Proposed selective encryption
    """
    print("\n" + "="*80)
    print("EXPERIMENT 7: Baseline Comparison (Section V baselines)")
    print("="*80)
    print(f"\n  {'Method':<25} {'E_exp':<10} {'E_enc':<10} {'Enc ratio':<12} {'Latency (ms)':<14} {'P95 (ms)'}")
    print("  " + "-"*75)

    # Baseline 1: No encryption (α=0, Poisson arrivals)
    sim_noenc = PipelineSimulation(seed=42)
    sim_noenc.run_poisson_arrivals(
        n_requests=100,
        arrival_rate_per_sec=2.0,
        request_factory=lambda i: make_request(i, alpha=0.0, domain="healthcare"),
    )
    no_enc = sim_noenc.results

    # Baseline 2: Full encryption (α=1.0, Poisson arrivals)
    sim_fullenc = PipelineSimulation(seed=42)
    sim_fullenc.run_poisson_arrivals(
        n_requests=100,
        arrival_rate_per_sec=2.0,
        request_factory=lambda i: make_request(i, alpha=1.0, domain="healthcare"),
    )
    full_enc = sim_fullenc.results

    # Baseline 3: Proposed selective encryption (α=0.5, Poisson arrivals)
    sim = PipelineSimulation(seed=42)
    sim.run_poisson_arrivals(
        n_requests=100,
        arrival_rate_per_sec=2.0,
        request_factory=lambda i: make_request(i, alpha=0.5, domain="healthcare"),
    )
    selective = sim.results

    for label, results in [
        ("No encryption (α=0)",     no_enc),
        ("Full encryption (α=1)",   full_enc),
        ("Selective (α=0.5, ours)", selective),
    ]:
        s = summarize(results)
        if s:
            avg_eexp = sum(r.eexp for r in results) / len(results)
            avg_eenc = sum(r.eenc for r in results) / len(results)
            avg_enc  = sum(r.encryption_ratio for r in results) / len(results)
            print(f"  {label:<25} {avg_eexp:<10.2f} {avg_eenc:<10.2f} "
                  f"{avg_enc:<12.3f} {s['latency_mean']:<14.1f} {s['latency_p95']:.1f}")


# ═════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS TO JSON (for plotting / paper tables)
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
            "eexp": r.eexp,
            "eenc": r.eenc,
            "alpha": r.alpha,
            "authorized": r.authorized,
            "t_detection": r.t_detection,
            "t_encryption": r.t_encryption,
            "t_zkp_gen": r.t_zkp_gen,
            "t_blockchain": r.t_blockchain,
            "t_tee": r.t_tee,
            "t_total": r.t_total,
            "wait_gateway": r.wait_gateway,
            "wait_blockchain": r.wait_blockchain,
            "wait_tee": r.wait_tee,
        })
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  ✓ Results saved to {filename}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*80)
    print("  ZKP-BLOCKCHAIN LLM PRIVACY PIPELINE — SimPy Simulation")
    print("  Timing calibrated from: RFC 7539 | zkSNARK benchmarks | Hyperledger Fabric")
    print("█"*80)

    experiment_single_trace()
    experiment_concurrent_users()
    experiment_poisson_arrivals()
    pareto_data = experiment_alpha_sweep()
    experiment_scalability()
    experiment_adversarial()
    experiment_domain_comparison()
    experiment_baseline_comparison()

    # Save alpha sweep results
    with open("alpha_sweep_results.json", "w") as f:
        json.dump(pareto_data, f, indent=2)
    print("\n  ✓ Alpha sweep saved to alpha_sweep_results.json")

    # Save full results for plotting
    sim_final = PipelineSimulation(seed=42)
    sim_final.run_poisson_arrivals(
        n_requests=500,
        arrival_rate_per_sec=2.0,
        request_factory=lambda i: make_request(i, domain="healthcare"),
    )
    save_results_json(sim_final.results, "simulation_results.json")

    print("\n" + "█"*80)
    print("  Done. Use simulation_results.json for plotting paper figures.")
    print("█"*80)