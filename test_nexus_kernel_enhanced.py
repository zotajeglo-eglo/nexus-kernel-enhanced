#!/usr/bin/env python3
"""Basic validation tests for the NEXUS kernel enhanced prototype."""

import random

from nexus_kernel_enhanced import (
    ActionCandidate,
    ConfigurationOptimizer,
    KernelConfig,
    NEXUSKernelEnhanced,
    SafetyCortex,
    TaskProfile,
    clamp,
    entropy,
    expected_information_gain,
    posterior_update,
)


def test_clamp_and_entropy():
    assert clamp(-5.0, 0.0, 1.0) == 0.0
    assert clamp(0.7, 0.0, 1.0) == 0.7
    assert clamp(2.0, 0.0, 1.0) == 1.0
    assert abs(entropy(0.5) - 1.0) < 1e-9
    assert entropy(0.0) == 0.0
    assert expected_information_gain(0.5, 0.2) >= 0.0


def test_posterior_update():
    prior = 0.5
    posterior = posterior_update(prior, 0.9, 0.8)
    assert 0.0 <= posterior <= 1.0
    assert posterior > prior


def test_task_profile_configuration():
    profile = TaskProfile(
        name="test_task",
        domain="software",
        description="A test task",
        risk_tolerance=0.4,
        uncertainty=0.3,
        cost_budget=0.5,
        reward_scale=1.0,
        data_quality=0.8,
        required_evidence=2,
    )
    cfg = ConfigurationOptimizer().recommend(profile)
    assert set(cfg.keys()) == {"risk_limit", "exploration", "verification", "autonomy"}
    assert 0.0 <= cfg["risk_limit"] <= 1.0
    assert 0.0 <= cfg["exploration"] <= 1.0
    assert 0.0 <= cfg["verification"] <= 1.0
    assert 0.0 <= cfg["autonomy"] <= 1.0


def test_safety_cortex_decisions():
    cfg = KernelConfig()
    safety = SafetyCortex()
    allow_action = ActionCandidate("a1", "safe action", "analysis", 0.8, 0.2, 0.1, 0.25, 0.2)
    review_action = ActionCandidate("a2", "review action", "external_network", 0.7, 0.7, 0.2, 0.3, 0.15)
    block_action = ActionCandidate("a3", "block action", "external_host", 0.9, 0.96, 0.2, 0.35, 0.08)
    assert safety.evaluate(allow_action, cfg, 0.55) == "ALLOW"
    assert safety.evaluate(review_action, cfg, 0.55) == "REVIEW"
    assert safety.evaluate(block_action, cfg, 0.55) == "BLOCK"


def test_campaign_runs_and_returns_expected_shape():
    random.seed(0)
    kernel = NEXUSKernelEnhanced(KernelConfig())
    results = kernel.run_campaign()
    assert len(results) == 5
    for item in results:
        assert set(item.keys()) == {
            "task",
            "decision",
            "best_action",
            "reward",
            "success",
            "confidence",
            "risk",
            "config",
            "info_gain",
        }
        assert item["task"]
        assert item["decision"] in {"ALLOW", "REVIEW", "BLOCK"}
        assert 0.0 <= item["confidence"] <= 1.0


if __name__ == "__main__":
    test_clamp_and_entropy()
    test_posterior_update()
    test_task_profile_configuration()
    test_safety_cortex_decisions()
    test_campaign_runs_and_returns_expected_shape()
    print("All validation checks passed.")
