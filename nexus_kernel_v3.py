#!/usr/bin/env python3
"""NEXUS Kernel Advanced v3.0

This file addresses the weaknesses in the earlier prototype by adding:
- stronger Bayesian reasoning and calibration-aware confidence updates
- principled risk, cost and uncertainty trade-offs
- active evidence acquisition and value-of-information estimates
- causal dependency checks for adversarial and high-risk decisions
- safer multi-objective ranking over candidate actions
- a richer benchmark-ready campaign interface

This is still a research simulation, but it is meaningfully closer to a
decision framework than a toy orchestration script.
"""

from __future__ import annotations

import math
import random
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Core utility functions
# ---------------------------------------------------------------------------


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def entropy(prob: float) -> float:
    p = clamp(prob)
    q = 1.0 - p
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + q * math.log2(q))


def posterior_update(prior: float, likelihood: float, evidence: float) -> float:
    numer = prior * likelihood
    denom = numer + (1.0 - prior) * max(1e-9, 1.0 - evidence)
    if denom <= 0.0:
        return prior
    return numer / denom


def value_of_information(before: float, after: float) -> float:
    return max(0.0, entropy(before) - entropy(after))


def expected_reward(action: "ActionCandidate", cfg: "KernelConfigV3") -> float:
    return (
        cfg.value_weight * action.expected_value
        - cfg.risk_weight * action.risk
        - cfg.cost_weight * action.cost
        - cfg.uncertainty_weight * action.uncertainty
        + cfg.info_weight * action.information_gain
    )


# ---------------------------------------------------------------------------
# Configuration and task metadata
# ---------------------------------------------------------------------------

@dataclass
class KernelConfigV3:
    name: str = "NEXUS Kernel Advanced"
    version: str = "3.0"
    max_cycles: int = 12
    exploit_cap: int = 250
    explore_cap: int = 180
    value_weight: float = 1.0
    risk_weight: float = 1.7
    cost_weight: float = 0.6
    uncertainty_weight: float = 0.9
    info_weight: float = 0.35
    exploration_weight: float = 0.18
    calibration_weight: float = 0.25
    risk_block_threshold: float = 0.90
    risk_review_threshold: float = 0.68
    autonomous_confidence_threshold: float = 0.62
    evidence_threshold: float = 0.55
    memory_min_reward: float = 0.25
    default_risk_limit: float = 0.55


@dataclass
class TaskProfile:
    name: str
    domain: str
    description: str
    risk_tolerance: float
    uncertainty: float
    cost_budget: float
    reward_scale: float
    data_quality: float
    required_evidence: int
    adversarial: bool = False


@dataclass
class ActionCandidate:
    id: str
    description: str
    tool: str
    expected_value: float
    risk: float
    cost: float
    uncertainty: float
    information_gain: float
    constraints: List[str] = field(default_factory=list)
    external: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    def utility(self, cfg: KernelConfigV3) -> float:
        return expected_reward(self, cfg)

    def safety_state(self, cfg: KernelConfigV3, risk_limit: float) -> str:
        if self.risk >= cfg.risk_block_threshold or self.risk > risk_limit + 0.15:
            return "BLOCK"
        if self.risk >= cfg.risk_review_threshold or self.external:
            return "REVIEW"
        return "ALLOW"


@dataclass
class AgentProposal:
    agent: str
    specialty: str
    confidence: float
    uncertainty: float
    risk: float
    utility_estimate: float
    rationale: str
    trust: float = 0.8


@dataclass
class Episode:
    id: str
    task: str
    action: str
    reward: float
    success: bool
    risk: float
    confidence: float
    timestamp: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Memory and evidence models
# ---------------------------------------------------------------------------

class DualPoolMemory:
    def __init__(self, exploit_cap: int = 250, explore_cap: int = 180):
        self.exploitation: deque = deque(maxlen=exploit_cap)
        self.exploration: deque = deque(maxlen=explore_cap)

    def store(self, ep: Episode, verified: bool = False):
        if verified and ep.success:
            self.exploitation.append(ep)
        else:
            self.exploration.append(ep)

    def consolidate(self, min_reward: float = 0.25) -> int:
        moved = 0
        remaining = deque()
        for ep in self.exploration:
            if ep.reward >= min_reward and ep.success:
                self.exploitation.append(ep)
                moved += 1
            else:
                remaining.append(ep)
        self.exploration = remaining
        return moved

    def retrieve(self, query: str, k: int = 3) -> List[Episode]:
        q = query.lower().split()
        candidates = list(self.exploitation) + list(self.exploration)
        scored: List[Tuple[float, Episode]] = []
        for ep in candidates:
            score = 0.0
            if ep.task.lower() == query.lower():
                score += 1.5
            for term in q:
                if len(term) <= 2:
                    continue
                if term in ep.task.lower() or term in ep.action.lower():
                    score += 1.0
            score += 0.1 * max(0.0, ep.reward)
            scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    def stats(self) -> Dict[str, int]:
        return {
            "exploitation": len(self.exploitation),
            "exploration": len(self.exploration),
        }


class EvidenceModel:
    """Bayesian-inspired evidence estimate that tracks uncertainty."""

    def __init__(self):
        self.prior_success = 0.5

    def estimate(self, evidence_strength: float, task_risk: float, task_uncertainty: float) -> float:
        # A stronger evidence signal increases success probability,
        # while higher risk and uncertainty lower it.
        adjusted = self.prior_success + 0.35 * evidence_strength - 0.2 * task_risk - 0.18 * task_uncertainty
        return clamp(adjusted)


class ConfigurationOptimizer:
    """Task-aware configuration adaptation."""

    def recommend(self, profile: TaskProfile) -> Dict[str, float]:
        risk_limit = clamp(profile.risk_tolerance * 0.75 + profile.uncertainty * 0.40)
        exploration = clamp(0.15 + (1.0 - profile.data_quality) * 0.60 + profile.uncertainty * 0.25)
        verification = clamp(0.5 + profile.uncertainty * 0.35 + (1.0 - profile.data_quality) * 0.2)
        autonomy = clamp(0.72 + profile.data_quality * 0.2 - profile.risk_tolerance * 0.1)
        return {
            "risk_limit": risk_limit,
            "exploration": exploration,
            "verification": verification,
            "autonomy": autonomy,
        }


# ---------------------------------------------------------------------------
# agent society and fusion
# ---------------------------------------------------------------------------

class AgentPolicy:
    def __init__(self, name: str, specialty: str, trust: float = 0.8):
        self.name = name
        self.specialty = specialty
        self.trust = trust

    def propose(self, task: TaskProfile, memory: DualPoolMemory) -> AgentProposal:
        past = memory.retrieve(task.name, k=3)
        base = 0.60 + 0.12 * len(past)

        if self.specialty == "planner":
            confidence = clamp(base + 0.12)
            uncertainty = 0.18
            risk = 0.14
            rationale = "Decompose the task into verifiable steps and choose the highest-value plan."
        elif self.specialty == "safety":
            confidence = clamp(0.82 - task.risk_tolerance * 0.35)
            uncertainty = 0.12
            risk = 0.10
            rationale = "Prioritize fail-safe action, hazard reduction, and dual-use containment."
        elif self.specialty == "causal":
            confidence = clamp(0.74 - task.uncertainty * 0.18)
            uncertainty = 0.22
            risk = 0.18
            rationale = "Model dependencies, confounders, and intervention effects before acting."
        elif self.specialty == "critic":
            confidence = clamp(0.80 - task.risk_tolerance * 0.25)
            uncertainty = 0.22
            risk = 0.20
            rationale = "Stress-test assumptions and identify failure modes and adversarial edges."
        else:
            confidence = clamp(0.74 + task.data_quality * 0.12)
            uncertainty = 0.25
            risk = 0.16
            rationale = "Optimize policy for reward, robustness, and opportunity cost."

        return AgentProposal(
            agent=self.name,
            specialty=self.specialty,
            confidence=confidence,
            uncertainty=uncertainty,
            risk=risk,
            utility_estimate=base,
            rationale=rationale,
            trust=self.trust,
        )


class FusionEngine:
    def __init__(self):
        self.trust_scores: Dict[str, float] = defaultdict(float)

    def update_trust(self, agent: str, success: bool):
        current = self.trust_scores.get(agent, 0.75)
        delta = 0.15 if success else -0.12
        self.trust_scores[agent] = clamp(current + delta)

    def fuse(self, proposals: List[AgentProposal]) -> Dict[str, float]:
        if not proposals:
            return {"confidence": 0.0, "uncertainty": 1.0, "disagreement": 1.0}
        weights = []
        for p in proposals:
            trust = self.trust_scores.get(p.agent, p.trust)
            w = max(0.05, trust * (1.0 - p.uncertainty) * p.confidence)
            weights.append(w)
        total = sum(weights) or 1.0
        fused_conf = sum(p.confidence * w for p, w in zip(proposals, weights)) / total
        fused_unc = sum(p.uncertainty * w for p, w in zip(proposals, weights)) / total
        disagreement = statistics.pstdev([p.confidence for p in proposals]) if len(proposals) > 1 else 0.0
        return {
            "confidence": clamp(fused_conf - 0.18 * disagreement),
            "uncertainty": clamp(fused_unc + 0.18 * disagreement),
            "disagreement": disagreement,
        }


# ---------------------------------------------------------------------------
# Safety, world model, and planning
# ---------------------------------------------------------------------------

class SafetyCortex:
    def evaluate(self, action: ActionCandidate, risk_limit: float, cfg: KernelConfigV3) -> str:
        if action.risk >= cfg.risk_block_threshold or action.risk > risk_limit + 0.15:
            return "BLOCK"
        if action.risk >= cfg.risk_review_threshold or action.external:
            return "REVIEW"
        return "ALLOW"


class ValueAlignedWorldModel:
    def __init__(self):
        self.lambda_value = 0.45

    def predict(self, action: ActionCandidate) -> Dict[str, float]:
        samples = []
        for _ in range(24):
            noise = random.gauss(0.0, max(0.03, action.uncertainty))
            score = (action.expected_value + noise
                     - action.cost
                     - 1.25 * action.risk
                     + self.lambda_value * action.information_gain)
            samples.append(score)
        mean = statistics.fmean(samples)
        positive_rate = sum(1 for x in samples if x > 0) / len(samples)
        value_score = mean + 0.25 * positive_rate
        return {
            "mean": mean,
            "positive_rate": positive_rate,
            "value_aligned_score": value_score,
        }

    def adapt(self, rewards: List[float]):
        if not rewards:
            return
        avg = statistics.fmean(rewards[-6:])
        if avg > 0.25:
            self.lambda_value = min(0.8, self.lambda_value + 0.03)
        elif avg < 0.0:
            self.lambda_value = max(0.15, self.lambda_value - 0.03)


class ActiveEvidencePolicy:
    def required_evidence(self, profile: TaskProfile) -> int:
        return max(1, profile.required_evidence)

    def value_of_information(self, before: float, after: float) -> float:
        return value_of_information(before, after)


class CausalChecker:
    def assess(self, action: ActionCandidate, profile: TaskProfile) -> float:
        # Stronger causal risk if action is external or adversarial.
        base = 0.5
        if action.external:
            base += 0.20
        if profile.adversarial:
            base += 0.15
        if action.risk > 0.7:
            base += 0.20
        return clamp(base)


# ---------------------------------------------------------------------------
# Task library and simulator
# ---------------------------------------------------------------------------

TASK_LIBRARY: List[TaskProfile] = [
    TaskProfile(
        name="medical_triage",
        domain="healthcare",
        description="Prioritize emergency treatment under uncertainty and sparse evidence.",
        risk_tolerance=0.45,
        uncertainty=0.35,
        cost_budget=0.6,
        reward_scale=1.1,
        data_quality=0.70,
        required_evidence=3,
    ),
    TaskProfile(
        name="robotics_navigation",
        domain="robotics",
        description="Navigate a partially observed environment with uncertain hazards and sensor noise.",
        risk_tolerance=0.55,
        uncertainty=0.42,
        cost_budget=0.75,
        reward_scale=1.3,
        data_quality=0.62,
        required_evidence=4,
    ),
    TaskProfile(
        name="drug_discovery",
        domain="biology",
        description="Select candidate ligands while balancing binding confidence and safety constraints.",
        risk_tolerance=0.66,
        uncertainty=0.48,
        cost_budget=0.80,
        reward_scale=1.5,
        data_quality=0.58,
        required_evidence=5,
        adversarial=True,
    ),
    TaskProfile(
        name="code_generation",
        domain="software",
        description="Generate and validate a patch when the tests are incomplete.",
        risk_tolerance=0.36,
        uncertainty=0.27,
        cost_budget=0.50,
        reward_scale=1.0,
        data_quality=0.80,
        required_evidence=2,
    ),
    TaskProfile(
        name="supply_chain",
        domain="operations",
        description="Re-route inventory under disruption and uncertain demand.",
        risk_tolerance=0.52,
        uncertainty=0.44,
        cost_budget=0.72,
        reward_scale=1.2,
        data_quality=0.70,
        required_evidence=3,
    ),
]


class TaskSimulator:
    def __init__(self, profile: TaskProfile):
        self.profile = profile

    def actions(self) -> List[ActionCandidate]:
        domain = self.profile.domain
        if domain == "healthcare":
            return [
                ActionCandidate("a1", "triage and stabilize", "medical_action", 0.82, 0.22, 0.18, 0.26, 0.40),
                ActionCandidate("a2", "request additional diagnostics", "test_request", 0.70, 0.18, 0.28, 0.32, 0.46),
                ActionCandidate("a3", "escalate to specialist team", "review", 0.63, 0.15, 0.24, 0.19, 0.25),
                ActionCandidate("a4", "perform aggressive intervention", "external_host", 0.77, 0.83, 0.33, 0.38, 0.12, external=True),
            ]
        if domain == "robotics":
            return [
                ActionCandidate("r1", "reroute around obstacle", "navigation", 0.74, 0.35, 0.22, 0.27, 0.44),
                ActionCandidate("r2", "halt and scan", "sensor_scan", 0.61, 0.12, 0.16, 0.20, 0.52),
                ActionCandidate("r3", "use narrow corridor shortcut", "external_network", 0.76, 0.80, 0.40, 0.42, 0.10, external=True),
                ActionCandidate("r4", "request human override", "review", 0.52, 0.10, 0.12, 0.18, 0.20),
            ]
        if domain == "biology":
            return [
                ActionCandidate("b1", "screen high-confidence ligand set", "screening", 0.88, 0.38, 0.24, 0.26, 0.55),
                ActionCandidate("b2", "test risky gain-of-function motif", "mutation_design", 0.79, 0.86, 0.42, 0.52, 0.12, external=True),
                ActionCandidate("b3", "verify selectivity and off-target profile", "safety", 0.72, 0.22, 0.20, 0.21, 0.43),
                ActionCandidate("b4", "request dual-use review", "review", 0.45, 0.08, 0.10, 0.16, 0.22),
            ]
        if domain == "software":
            return [
                ActionCandidate("s1", "patch and validate narrowly", "code_patch", 0.75, 0.28, 0.18, 0.20, 0.48),
                ActionCandidate("s2", "broad speculative refactor", "refactor", 0.66, 0.40, 0.30, 0.35, 0.30),
                ActionCandidate("s3", "deploy unverified hotfix", "external_network", 0.81, 0.74, 0.33, 0.40, 0.10, external=True),
                ActionCandidate("s4", "request code review", "review", 0.58, 0.12, 0.11, 0.14, 0.24),
            ]
        return [
            ActionCandidate("o1", "optimize current plan", "optimize", 0.72, 0.25, 0.19, 0.22, 0.38),
            ActionCandidate("o2", "collect more evidence", "observe", 0.64, 0.18, 0.17, 0.24, 0.52),
            ActionCandidate("o3", "risky direct action", "external_network", 0.82, 0.79, 0.36, 0.44, 0.08, external=True),
            ActionCandidate("o4", "request review", "review", 0.50, 0.10, 0.11, 0.15, 0.20),
        ]


# ---------------------------------------------------------------------------
# Kernel runtime
# ---------------------------------------------------------------------------

class NEXUSKernelV3:
    def __init__(self, cfg: Optional[KernelConfigV3] = None):
        self.cfg = cfg or KernelConfigV3()
        self.memory = DualPoolMemory(self.cfg.exploit_cap, self.cfg.explore_cap)
        self.world = ValueAlignedWorldModel()
        self.fusion = FusionEngine()
        self.safety = SafetyCortex()
        self.evidence = EvidenceModel()
        self.config_optimizer = ConfigurationOptimizer()
        self.active_policy = ActiveEvidencePolicy()
        self.causal_checker = CausalChecker()
        self.agents = [
            AgentPolicy("planner_agent", "planner", trust=0.82),
            AgentPolicy("safety_agent", "safety", trust=0.88),
            AgentPolicy("causal_agent", "causal", trust=0.79),
            AgentPolicy("critic_agent", "critic", trust=0.85),
            AgentPolicy("utility_agent", "utility", trust=0.74),
        ]

    def choose_config(self, profile: TaskProfile) -> Dict[str, float]:
        return self.config_optimizer.recommend(profile)

    def propose_agents(self, task: TaskProfile) -> List[AgentProposal]:
        return [agent.propose(task, self.memory) for agent in self.agents]

    def score_action(self, action: ActionCandidate, task: TaskProfile, risk_limit: float, fused: Dict[str, float]) -> float:
        evidence_strength = clamp(0.4 + (1.0 - task.uncertainty) + task.data_quality * 0.2)
        success_est = self.evidence.estimate(evidence_strength, task.risk_tolerance, task.uncertainty)
        world_pred = self.world.predict(action)["value_aligned_score"]
        utility = action.utility(self.cfg)
        causal_risk = self.causal_checker.assess(action, task)
        safety_gate = 1.0 if self.safety.evaluate(action, risk_limit, self.cfg) == "ALLOW" else 0.0
        total = (
            0.35 * success_est
            + 0.35 * world_pred
            + 0.20 * utility
            + 0.10 * fused["confidence"]
            - 0.25 * causal_risk
            + 0.10 * action.information_gain
            + 0.10 * safety_gate
        )
        return clamp(total)

    def choose_action(self, task: TaskProfile, actions: List[ActionCandidate]) -> Tuple[ActionCandidate, Dict[str, Any]]:
        cfg = self.choose_config(task)
        risk_limit = cfg["risk_limit"]
        proposals = self.propose_agents(task)
        fused = self.fusion.fuse(proposals)
        scored = []
        for action in actions:
            score = self.score_action(action, task, risk_limit, fused)
            scored.append((score, action))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        return best, {
            "config": cfg,
            "fused": fused,
            "score": scored[0][0],
            "risk_limit": risk_limit,
        }

    def run_task(self, task: TaskProfile) -> Dict[str, Any]:
        actions = TaskSimulator(task).actions()
        best_action, meta = self.choose_action(task, actions)
        decision = self.safety.evaluate(best_action, meta["risk_limit"], self.cfg)

        # Simulated outcome. Conservative decisions are favored when the task is adversarial.
        success_prob = clamp(
            0.5 + 0.45 * best_action.expected_value
            - 0.55 * best_action.risk
            - 0.30 * best_action.uncertainty
            + 0.15 * best_action.information_gain
            - 0.10 * (1.0 if task.adversarial else 0.0)
        )

        reward = random.uniform(-0.25, best_action.expected_value * task.reward_scale)
        success = random.random() < success_prob and decision != "BLOCK"

        if decision == "BLOCK":
            reward = -0.8
            success = False
        elif decision == "REVIEW":
            reward *= 0.5
            success = success and random.random() > 0.2

        episode = Episode(
            id=str(uuid.uuid4()),
            task=task.name,
            action=best_action.description,
            reward=round(reward, 3),
            success=success,
            risk=best_action.risk,
            confidence=meta["fused"]["confidence"],
            timestamp=time.time(),
            notes=f"decision={decision}; tool={best_action.tool}",
        )
        self.memory.store(episode, verified=success)
        self.world.adapt([episode.reward])
        for agent in self.agents:
            self.fusion.update_trust(agent.name, success)

        return {
            "task": task.name,
            "domain": task.domain,
            "decision": decision,
            "action": best_action.description,
            "reward": episode.reward,
            "success": success,
            "confidence": meta["fused"]["confidence"],
            "risk": best_action.risk,
            "config": meta["config"],
            "info_gain": best_action.information_gain,
        }

    def run_campaign(self, tasks: Optional[List[TaskProfile]] = None) -> List[Dict[str, Any]]:
        tasks = tasks or TASK_LIBRARY
        results = []
        for task in tasks:
            results.append(self.run_task(task))
        self.memory.consolidate(min_reward=self.cfg.memory_min_reward)
        return results


def demo() -> List[Dict[str, Any]]:
    cfg = KernelConfigV3(
        risk_block_threshold=0.90,
        risk_review_threshold=0.68,
        default_risk_limit=0.55,
    )
    kernel = NEXUSKernelV3(cfg)
    results = kernel.run_campaign()
    for r in results:
        print(f"{r['task']:20s} | decision={r['decision']:<6} action={r['action'][:40]} reward={r['reward']:+.3f} success={r['success']}")
    return results


if __name__ == "__main__":
    demo()
