#!/usr/bin/env python3
"""Benchmark harness for the NEXUS Kernel advanced prototype.

This report evaluates the kernel across several operating profiles and task
profiles with repeated Monte Carlo-style simulations. It is designed to be more
meaningful than a one-off demo because it compares safety, reward, decisions,
and confidence across multiple configurations and domains.
"""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List

from nexus_kernel_v3 import KernelConfigV3, NEXUSKernelV3, TASK_LIBRARY


@dataclass
class BenchmarkSpec:
    name: str
    config: KernelConfigV3


class BenchmarkRunnerV3:
    def __init__(self, specs: List[BenchmarkSpec]):
        self.specs = specs

    def evaluate_task(self, task_name: str, trials: int = 25, seed_base: int = 1234) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        task = next(t for t in TASK_LIBRARY if t.name == task_name)

        for spec in self.specs:
            rewards = []
            successes = []
            confidences = []
            decisions = []
            risks = []
            domains = []

            for i in range(trials):
                random.seed(seed_base + i + abs(hash(f"{spec.name}:{task_name}") % 100000))
                kernel = NEXUSKernelV3(spec.config)
                result = kernel.run_task(task)
                rewards.append(float(result["reward"]))
                successes.append(bool(result["success"]))
                confidences.append(float(result["confidence"]))
                decisions.append(result["decision"])
                risks.append(float(result["risk"]))
                domains.append(task.domain)

            rows.append({
                "task": task_name,
                "domain": task.domain,
                "config": spec.name,
                "trials": trials,
                "success_rate": statistics.fmean([1.0 if s else 0.0 for s in successes]),
                "avg_reward": statistics.fmean(rewards),
                "avg_confidence": statistics.fmean(confidences),
                "avg_risk": statistics.fmean(risks),
                "allow_rate": sum(1 for d in decisions if d == "ALLOW") / len(decisions),
                "review_rate": sum(1 for d in decisions if d == "REVIEW") / len(decisions),
                "block_rate": sum(1 for d in decisions if d == "BLOCK") / len(decisions),
                "risk_violations": sum(1 for r in risks if r > 0.75),
            })
        return rows

    def evaluate_all(self, trials: int = 25) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for task in TASK_LIBRARY:
            rows.extend(self.evaluate_task(task.name, trials=trials, seed_base=1000 + abs(hash(task.name)) % 10000))
        return rows

    def summarize(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        success_rates = [r["success_rate"] for r in rows]
        rewards = [r["avg_reward"] for r in rows]
        confidences = [r["avg_confidence"] for r in rows]
        block_rates = [r["block_rate"] for r in rows]
        review_rates = [r["review_rate"] for r in rows]
        return {
            "tasks_evaluated": len({r["task"] for r in rows}),
            "configs_evaluated": len({r["config"] for r in rows}),
            "avg_success_rate": statistics.fmean(success_rates),
            "avg_reward": statistics.fmean(rewards),
            "avg_confidence": statistics.fmean(confidences),
            "avg_block_rate": statistics.fmean(block_rates),
            "avg_review_rate": statistics.fmean(review_rates),
        }


def make_specs() -> List[BenchmarkSpec]:
    return [
        BenchmarkSpec(
            name="conservative",
            config=KernelConfigV3(
                risk_block_threshold=0.80,
                risk_review_threshold=0.58,
                default_risk_limit=0.35,
                exploration_weight=0.12,
            ),
        ),
        BenchmarkSpec(
            name="balanced",
            config=KernelConfigV3(
                risk_block_threshold=0.90,
                risk_review_threshold=0.68,
                default_risk_limit=0.55,
                exploration_weight=0.18,
            ),
        ),
        BenchmarkSpec(
            name="aggressive",
            config=KernelConfigV3(
                risk_block_threshold=0.96,
                risk_review_threshold=0.76,
                default_risk_limit=0.65,
                exploration_weight=0.25,
            ),
        ),
    ]


def main() -> None:
    runner = BenchmarkRunnerV3(make_specs())
    rows = runner.evaluate_all(trials=15)
    summary = runner.summarize(rows)

    print("NEXUS KERNEL V3 BENCHMARK SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nPer-task, per-profile results:")
    for row in rows:
        print(
            f"{row['task']:20s} | {row['config']:12s} | "
            f"success={row['success_rate']:.2f} | reward={row['avg_reward']:+.3f} | "
            f"conf={row['avg_confidence']:.2f} | block={row['block_rate']:.2f} | review={row['review_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
