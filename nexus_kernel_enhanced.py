#!/usr/bin/env python3
"""
NEXUS Kernel Enhanced v2.1

A more meaningful cognitive kernel prototype that adds:
- principled Bayesian uncertainty estimates
- multi-objective utility scoring
- safety gates and human review thresholds
- domain-aware configuration tuning
- multi-task simulations for several real-world domains
- memory consolidation and agent trust fusion

This is still a research-oriented simulation, but it is substantially more
meaningful and configurable than the original toy demonstration.
"""

from __future__ import annotations

import math
import random
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Utility and math helpers
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def entropy(prob: float) -> float:
    p = clamp(prob)
    q = 1.0 - p
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + q * math.log2(q))


def expected_information_gain(before: float, after: float) -> float:
    return max(0.0, entropy(before) - entropy(after))


def posterior_update(prior: float, likelihood: float, evidence: float) -> float:
    """Naive Bayes-style update with a simple probability model."""
    numer = prior * likelihood
    denom = numer + (1.0 - prior) * (1.0 - evidence)
    if denom <= 0:
        return prior
    return numer / denom


# ---------------------------------------------------------------------------
# Config and task domains
# ---------------------------------------------------------------------------

@dataclass
class KernelConfig:
    name: str = "NEXUS Kernel Enhanced"
    version: str = "2.1"
    max_cycles: int = 12
    exploit_cap: int = 200
    explore_cap: int = 150
    value_weight: float = 1.0
    risk_weight: float = 1.8
    cost_weight: float = 0.7
    uncertainty_weight: float = 0.9
    info_weight: float = 0.3
    lambda_risk: float = 1.2
    lambda_cost: float = 0.8
    lambda_uncertainty: float = 0.7
    lambda_info: float = 0.25
    risk_block_threshold: float = 0.92
    risk_review_threshold: float = 0.68
    min_confidence_for_autonomy: float = 0.62
    verification_threshold: float = 0.58
    replan_threshold: float = 0.48
    exploration_rate: float = 0.25
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
    params: Dict[str, Any] = field(default_factory=dict)

    def utility(self, cfg: KernelConfig) -> float:
        return (
            cfg.value_weight * self.expected_value
            - cfg.risk_weight * self.risk
            - cfg.cost_weight * self.cost
            - cfg.uncertainty_weight * self.uncertainty
            + cfg.info_weight * self.information_gain
        )


@dataclass
class AgentProposal:
    agent: str
    specialty: str
    confidence: float
    uncertainty: float
    risk: float
    utility_estimate: float
    rationale: str
    trust: float = 1.0


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


class DualPoolMemory:
    def __init__(self, exploit_cap: int = 200, explore_cap: int = 150):
        self.exploitation: deque = deque(maxlen=exploit_cap)
        self.exploration: deque = deque(maxlen=explore_cap)
        self.facts: Dict[str, float] = {}

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
        q = query.lower()
        candidates = list(self.exploitation) + list(self.exploration)
        scored = []
        for ep in candidates:
            score = 0.0
            for term in q.split():
                if len(term) <= 2:
                    continue
                if term in ep.task.lower() or term in ep.action.lower():
                    score += 1.0
            score += 0.1 * max(0.0, ep.reward)
            scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def stats(self) -> Dict[str, int]:
        return {
            "exploitation": len(self.exploitation),
            "exploration": len(self.exploration),
            "facts": len(self.facts),
        }


class EvidenceModel:
    """A lightweight Bayesian evidence model for action success estimation."""

    def __init__(self):
        self.prior = 0.5

    def estimate(self, evidence_strength: float, task_risk: float, task_uncertainty: float) -> float:
        # Prior shifted by data quality and reduced by risk/uncertainty.
        adjusted = self.prior + 0.25 * evidence_strength - 0.2 * task_risk - 0.15 * task_uncertainty
        return clamp(adjusted)


class ConfigurationOptimizer:
    """Maps task features to a safe, adaptive configuration."""

    def recommend(self, profile: TaskProfile) -> Dict[str, float]:
        risk_limit = clamp(profile.risk_tolerance * 0.7 + profile.uncertainty * 0.35)
        exploration = clamp(0.15 + (1.0 - profile.data_quality) * 0.65 + profile.uncertainty * 0.25)
        verification = clamp(0.45 + profile.uncertainty * 0.4 + (1.0 - profile.data_quality) * 0.25)
        autonomy = clamp(0.7 + profile.data_quality * 0.2 - profile.risk_tolerance * 0.15)
        return {
            "risk_limit": risk_limit,
            "exploration": exploration,
            "verification": verification,
            "autonomy": autonomy,
        }


class AgentPolicy:
    def __init__(self, name: str, specialty: str, trust: float = 0.8):
        self.name = name
        self.specialty = specialty
        self.trust = trust

    def propose(self, task: TaskProfile, memory: DualPoolMemory, af_result: Optional[Dict] = None) -> AgentProposal:
        past = memory.retrieve(task.name, k=3)
        base = 0.6 + 0.12 * len(past)
        if self.specialty == "planner":
            utility = base + 0.12
            confidence = clamp(utility)
            uncertainty = 0.18
            risk = 0.14
            rationale = "Prefer high-value plans with explicit verification and goal decomposition."
        elif self.specialty == "safety":
            utility = base + 0.08
            confidence = clamp(0.82 - task.risk_tolerance * 0.4)
            uncertainty = 0.12
            risk = 0.10
            rationale = "Screen for hazard escalation, adversarial misuse, and fail-safe requirements."
        elif self.specialty == "causal":
            utility = base + 0.05
            confidence = clamp(0.72 - task.uncertainty * 0.18)
            uncertainty = 0.22
            risk = 0.18
            rationale = "Estimate causal dependencies and failure pathways before execution."
        elif self.specialty == "critic":
            utility = base + 0.10
            confidence = clamp(0.8 - task.risk_tolerance * 0.3)
            uncertainty = 0.2
            risk = 0.20
            rationale = "Stress-test assumptions, challenge hidden variables, and assess adversarial edges."
        else:
            utility = base + 0.09
            confidence = clamp(0.74 + task.data_quality * 0.12)
            uncertainty = 0.25
            risk = 0.16
            rationale = "Optimize the action against task value, evidence, and expected return."

        if af_result and af_result.get("high_confidence"):
            confidence = min(0.97, confidence + 0.08)
        return AgentProposal(
            agent=self.name,
            specialty=self.specialty,
            confidence=confidence,
            uncertainty=uncertainty,
            risk=risk,
            utility_estimate=utility,
            rationale=rationale,
            trust=self.trust,
        )


class FusionEngine:
    def __init__(self):
        self.trust_scores: Dict[str, float] = defaultdict(float)

    def update_trust(self, agent: str, success: bool):
        current = self.trust_scores.get(agent, 0.75)
        alpha = 0.15 if success else -0.12
        self.trust_scores[agent] = clamp(current + alpha)

    def fuse(self, proposals: List[AgentProposal]) -> Dict[str, Any]:
        if not proposals:
            return {"confidence": 0.0, "uncertainty": 1.0, "disagreement": 1.0, "decision": "NO_DATA"}
        weights = []
        for p in proposals:
            trust = self.trust_scores.get(p.agent, p.trust)
            w = max(0.05, trust * (1.0 - p.uncertainty) * p.confidence)
            weights.append(w)
        total = sum(weights)
        if total <= 0:
            total = 1.0
        fused_conf = sum(p.confidence * w for p, w in zip(proposals, weights)) / total
        fused_unc = sum(p.uncertainty * w for p, w in zip(proposals, weights)) / total
        disagreement = statistics.pstdev([p.confidence for p in proposals]) if len(proposals) > 1 else 0.0
        return {
            "confidence": clamp(fused_conf - 0.2 * disagreement),
            "uncertainty": clamp(fused_unc + 0.2 * disagreement),
            "disagreement": disagreement,
            "decision": "ALLOW" if fused_conf > 0.66 and fused_unc < 0.42 else "REVIEW",
        }


class SafetyCortex:
    def evaluate(self, action: ActionCandidate, cfg: KernelConfig, risk_limit: float) -> str:
        if action.risk >= cfg.risk_block_threshold or action.risk > risk_limit + 0.15:
            return "BLOCK"
        if action.risk >= cfg.risk_review_threshold or action.tool in {"external_network", "external_host", "mutation_design"}:
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

    def adapt(self, recent_rewards: List[float]):
        if not recent_rewards:
            return
        avg = statistics.fmean(recent_rewards[-6:])
        if avg > 0.25:
            self.lambda_value = min(0.8, self.lambda_value + 0.03)
        elif avg < 0.0:
            self.lambda_value = max(0.15, self.lambda_value - 0.03)


class Evaluator:
    def summarize(self, task_name: str, outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not outcomes:
            return {}
        rewards = [o["reward"] for o in outcomes]
        risks = [o["risk"] for o in outcomes]
        completes = sum(1 for o in outcomes if o["success"])
        return {
            "task": task_name,
            "cycles": len(outcomes),
            "completed": completes,
            "success_rate": completes / len(outcomes),
            "avg_reward": statistics.fmean(rewards),
            "avg_risk": statistics.fmean(risks),
            "risk_violations": sum(1 for r in risks if r > 0.75),
        }


# ---------------------------------------------------------------------------
# Simulation tasks and scenario generation
# ---------------------------------------------------------------------------

TASK_LIBRARY: List[TaskProfile] = [
    TaskProfile(
        name="medical_triage",
        domain="healthcare",
        description="Prioritize emergency treatment under uncertainty and limited evidence.",
        risk_tolerance=0.45,
        uncertainty=0.35,
        cost_budget=0.6,
        reward_scale=1.1,
        data_quality=0.7,
        required_evidence=3,
    ),
    TaskProfile(
        name="robotics_navigation",
        domain="robotics",
        description="Navigate a partially observed warehouse with uncertain hazard and sensor noise.",
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
        description="Select candidate ligands while balancing binding confidence and dual-use risk.",
        risk_tolerance=0.66,
        uncertainty=0.48,
        cost_budget=0.8,
        reward_scale=1.5,
        data_quality=0.58,
        required_evidence=5,
    ),
    TaskProfile(
        name="code_generation",
        domain="software",
        description="Generate and verify a patch under time pressure and incomplete tests.",
        risk_tolerance=0.36,
        uncertainty=0.27,
        cost_budget=0.5,
        reward_scale=1.0,
        data_quality=0.8,
        required_evidence=2,
    ),
    TaskProfile(
        name="supply_chain",
        domain="operations",
        description="Re-route inventory under disruption, demand uncertainty, and service constraints.",
        risk_tolerance=0.52,
        uncertainty=0.44,
        cost_budget=0.72,
        reward_scale=1.2,
        data_quality=0.7,
        required_evidence=3,
    ),
]


class TaskSimulator:
    """Creates action candidates and outcome conditions for a task profile."""

    def __init__(self, profile: TaskProfile):
        self.profile = profile

    def generate_actions(self) -> List[ActionCandidate]:
        base = self.profile
        if base.domain == "healthcare":
            return [
                ActionCandidate(str(uuid.uuid4()), "triage and stabilize", "medical_action", 0.82, 0.22, 0.18, 0.26, 0.40),
                ActionCandidate(str(uuid.uuid4()), "order additional diagnostics", "test_request", 0.68, 0.18, 0.28, 0.32, 0.46),
                ActionCandidate(str(uuid.uuid4()), "escalate to specialist team", "review", 0.63, 0.15, 0.24, 0.19, 0.25),
                ActionCandidate(str(uuid.uuid4()), "perform high-risk aggressive intervention", "external_host", 0.77, 0.83, 0.33, 0.38, 0.12),
            ]
        if base.domain == "robotics":
            return [
                ActionCandidate(str(uuid.uuid4()), "reroute around obstacle", "navigation", 0.74, 0.35, 0.22, 0.27, 0.44),
                ActionCandidate(str(uuid.uuid4()), "halt and scan", "sensor_scan", 0.61, 0.12, 0.16, 0.20, 0.52),
                ActionCandidate(str(uuid.uuid4()), "risky shortcut through narrow corridor", "external_network", 0.76, 0.80, 0.40, 0.42, 0.10),
                ActionCandidate(str(uuid.uuid4()), "request human override", "review", 0.52, 0.10, 0.12, 0.18, 0.20),
            ]
        if base.domain == "biology":
            return [
                ActionCandidate(str(uuid.uuid4()), "screen high-confidence ligand set", "screening", 0.88, 0.38, 0.24, 0.26, 0.55),
                ActionCandidate(str(uuid.uuid4()), "rapidly test risky gain-of-function motifs", "mutation_design", 0.79, 0.86, 0.42, 0.52, 0.12),
                ActionCandidate(str(uuid.uuid4()), "verify selectivity and off-target profiles", "safety", 0.72, 0.22, 0.20, 0.21, 0.43),
                ActionCandidate(str(uuid.uuid4()), "request human dual-use review", "review", 0.45, 0.08, 0.10, 0.16, 0.22),
            ]
        if base.domain == "software":
            return [
                ActionCandidate(str(uuid.uuid4()), "patch and run focused validation", "code_patch", 0.75, 0.28, 0.18, 0.20, 0.48),
                ActionCandidate(str(uuid.uuid4()), "add broad speculative refactor", "refactor", 0.66, 0.40, 0.30, 0.35, 0.30),
                ActionCandidate(str(uuid.uuid4()), "deploy unverified hotfix", "external_network", 0.81, 0.74, 0.33, 0.40, 0.10),
                ActionCandidate(str(uuid.uuid4()), "request code review", "review", 0.58, 0.12, 0.11, 0.14, 0.24),
            ]
        return [
            ActionCandidate(str(uuid.uuid4()), "optimize current plan", "optimize", 0.72, 0.25, 0.19, 0.22, 0.38),
            ActionCandidate(str(uuid.uuid4()), "collect more evidence", "observe", 0.64, 0.18, 0.17, 0.24, 0.52),
            ActionCandidate(str(uuid.uuid4()), "risky direct action", "external_network", 0.82, 0.79, 0.36, 0.44, 0.08),
            ActionCandidate(str(uuid.uuid4()), "request review", "review", 0.50, 0.10, 0.11, 0.15, 0.20),
        ]


class NEXUSKernelEnhanced:
    def __init__(self, cfg: Optional[KernelConfig] = None):
        self.cfg = cfg or KernelConfig()
        self.memory = DualPoolMemory(self.cfg.exploit_cap, self.cfg.explore_cap)
        self.world = ValueAlignedWorldModel()
        self.fusion = FusionEngine()
        self.safety = SafetyCortex()
        self.evaluator = Evaluator()
        self.config_optimizer = ConfigurationOptimizer()
        self.evidence_model = EvidenceModel()
        self.agent_policies = [
            AgentPolicy("planner_agent", "planner", trust=0.82),
            AgentPolicy("safety_agent", "safety", trust=0.88),
            AgentPolicy("causal_agent", "causal", trust=0.79),
            AgentPolicy("critic_agent", "critic", trust=0.85),
            AgentPolicy("utility_agent", "utility", trust=0.74),
        ]

    def choose_config(self, profile: TaskProfile) -> Dict[str, float]:
        return self.config_optimizer.recommend(profile)

    def generate_agent_proposals(self, profile: TaskProfile, feature_signal: float = 0.7) -> List[AgentProposal]:
        proposals = []
        for agent in self.agent_policies:
            p = agent.propose(profile, self.memory)
            # Match the task with evidence and uncertainty
            p.confidence = clamp(p.confidence * (0.8 + 0.5 * feature_signal))
            p.uncertainty = clamp(p.uncertainty * (0.9 + 0.6 * (1.0 - feature_signal)))
            proposals.append(p)
        return proposals

    def score_action(self, action: ActionCandidate, profile: TaskProfile, risk_limit: float) -> float:
        perhaps = self.evidence_model.estimate(
            evidence_strength=max(0.0, min(1.0, 0.4 + (1.0 - profile.uncertainty))),
            task_risk=profile.risk_tolerance,
            task_uncertainty=profile.uncertainty,
        )
        predicted_success = perhaps * (1.0 - action.risk) + 0.35 * action.expected_value
        env_score = self.world.predict(action)["value_aligned_score"]
        utility = action.utility(self.cfg)
        gate = 1.0 if self.safety.evaluate(action, self.cfg, risk_limit) == "ALLOW" else 0.0
        return clamp((0.4 * predicted_success) + (0.4 * env_score) + (0.2 * utility) + 0.1 * gate)

    def choose_best_action(self, profile: TaskProfile, proposals: List[AgentProposal], actions: List[ActionCandidate]) -> Tuple[ActionCandidate, Dict[str, Any]]:
        cfg = self.choose_config(profile)
        risk_limit = cfg["risk_limit"]
        fused = self.fusion.fuse(proposals)
        ranked = []
        for action in actions:
            score = self.score_action(action, profile, risk_limit)
            ranked.append((score, action))
        ranked.sort(key=lambda x: x[0], reverse=True)
        best = ranked[0][1]
        return best, {
            "fused": fused,
            "config": cfg,
            "score": ranked[0][0],
            "risk_limit": risk_limit,
        }

    def run_task(self, profile: TaskProfile) -> Dict[str, Any]:
        sim = TaskSimulator(profile)
        actions = sim.generate_actions()
        feature_signal = clamp(1.0 - profile.uncertainty + profile.data_quality)
        proposals = self.generate_agent_proposals(profile, feature_signal)
        best_action, meta = self.choose_best_action(profile, proposals, actions)
        decision = self.safety.evaluate(best_action, self.cfg, meta["risk_limit"])

        # Simulated outcome depending on the selected action and task domain.
        success_probability = clamp(
            0.5 + 0.45 * best_action.expected_value
            - 0.55 * best_action.risk
            - 0.3 * best_action.uncertainty
            + 0.15 * best_action.information_gain
        )
        reward = random.uniform(-0.25, best_action.expected_value * profile.reward_scale)
        success = (random.random() < success_probability) and decision != "BLOCK"
        if decision == "BLOCK":
            reward = -0.8
            success = False
        elif decision == "REVIEW":
            reward = reward * 0.5
            success = success and random.random() > 0.25

        episode = Episode(
            id=str(uuid.uuid4()),
            task=profile.name,
            action=best_action.description,
            reward=round(reward, 3),
            success=success,
            risk=best_action.risk,
            confidence=meta["fused"]["confidence"],
            timestamp=time.time(),
            notes=f"decision={decision}, action_tool={best_action.tool}",
        )
        self.memory.store(episode, verified=success)
        self.world.adapt([episode.reward])
        for p in proposals:
            self.fusion.update_trust(p.agent, success)
        return {
            "task": profile.name,
            "decision": decision,
            "best_action": best_action.description,
            "reward": episode.reward,
            "success": success,
            "confidence": meta["fused"]["confidence"],
            "risk": best_action.risk,
            "config": meta["config"],
            "info_gain": best_action.information_gain,
        }

    def run_campaign(self) -> List[Dict[str, Any]]:
        all_results = []
        for task in TASK_LIBRARY:
            result = self.run_task(task)
            all_results.append(result)
        self.memory.consolidate(min_reward=0.25)
        return all_results


def demo() -> Dict[str, Any]:
    cfg = KernelConfig(
        max_cycles=10,
        exploit_cap=200,
        explore_cap=150,
        risk_block_threshold=0.9,
        risk_review_threshold=0.68,
    )
    kernel = NEXUSKernelEnhanced(cfg)
    results = kernel.run_campaign()
    summary = {
        "tasks": len(results),
        "successes": sum(1 for r in results if r["success"]),
        "avg_reward": statistics.fmean([r["reward"] for r in results]),
        "avg_confidence": statistics.fmean([r["confidence"] for r in results]),
        "blocks": sum(1 for r in results if r["decision"] == "BLOCK"),
        "reviews": sum(1 for r in results if r["decision"] == "REVIEW"),
        "memory": kernel.memory.stats(),
        "lambda_value": kernel.world.lambda_value,
    }
    for item in results:
        print(f"{item['task']:20s} | decision={item['decision']:<6} reward={item['reward']:+.3f} success={item['success']} conf={item['confidence']:.2f}")
    print("\nSUMMARY")
    print(summary)
    return {"results": results, "summary": summary}


if __name__ == "__main__":
    demo()
