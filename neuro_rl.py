"""NeuroRL — Reinforcement Learning on LLM prompts using brain signals.

The core innovation: instead of fine-tuning model weights, we optimize the
PROMPT's style parameters using the user's brain response as reward.

Algorithm: Contextual Thompson Sampling bandit.
  - Context: brain state features (12-dim)
  - Action: style parameter vector (8-dim continuous)
  - Reward: scalar from brain response (engagement + valence)

The bandit maintains a posterior over style-parameter preferences conditioned
on brain state, and samples from it to balance exploration/exploitation.
"""

import json
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config import (
    STYLE_PARAMS, STYLE_DIM, BRAIN_STATE_DIM,
    RL_LEARNING_RATE, RL_EXPLORATION,
    REWARD_WEIGHTS, CLENCH_BONUS, ERRP_PENALTY,
)


@dataclass
class StyleVector:
    formality: float = 0.3
    enthusiasm: float = 0.6
    verbosity: float = 0.4
    empathy: float = 0.7
    humor: float = 0.3
    assertiveness: float = 0.5
    emotional_expressiveness: float = 0.6
    warmth: float = 0.7

    def to_array(self):
        return np.array([getattr(self, p) for p in STYLE_PARAMS], dtype=np.float32)

    @classmethod
    def from_array(cls, arr):
        return cls(**{p: float(np.clip(arr[i], 0, 1)) for i, p in enumerate(STYLE_PARAMS)})

    def to_prompt_instructions(self):
        parts = []
        if self.formality > 0.7:
            parts.append("Use formal, professional language.")
        elif self.formality < 0.3:
            parts.append("Be very casual and informal.")

        if self.enthusiasm > 0.7:
            parts.append("Be enthusiastic and energetic.")
        elif self.enthusiasm < 0.3:
            parts.append("Be calm and measured.")

        if self.verbosity > 0.7:
            parts.append("Give detailed, thorough responses.")
        elif self.verbosity < 0.3:
            parts.append("Keep responses very brief — a few words or one sentence max.")
        else:
            parts.append("Keep responses concise — 1-2 sentences.")

        if self.empathy > 0.7:
            parts.append("Show deep empathy and emotional understanding.")
        elif self.empathy < 0.3:
            parts.append("Be direct and matter-of-fact.")

        if self.humor > 0.6:
            parts.append("Include gentle humor or playfulness when appropriate.")

        if self.assertiveness > 0.7:
            parts.append("Be confident and assertive.")
        elif self.assertiveness < 0.3:
            parts.append("Be gentle and tentative.")

        if self.emotional_expressiveness > 0.7:
            parts.append("Express emotions openly.")

        if self.warmth > 0.7:
            parts.append("Be warm, caring, and personal.")
        elif self.warmth < 0.3:
            parts.append("Be professional and neutral.")

        return " ".join(parts)


@dataclass
class RLExperience:
    brain_context: np.ndarray
    style_used: np.ndarray
    reward: float
    timestamp: float
    intent: str = ""
    jaw_clench: bool = False
    error_detected: bool = False


class NeuroRLEngine:
    """Contextual bandit that learns style preferences from brain signals.

    Uses REINFORCE-style policy gradient: maintain a mean + variance per style
    parameter, sample from Gaussian, update mean toward high-reward samples.
    """

    def __init__(self, brain_dim=BRAIN_STATE_DIM, style_dim=STYLE_DIM):
        self.brain_dim = brain_dim
        self.style_dim = style_dim

        self._current_style = StyleVector()

        # Policy parameters: mean and log-variance per style dim
        # Mean is what we believe the user prefers
        self._mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                             dtype=np.float64)
        self._log_var = np.zeros(style_dim, dtype=np.float64)

        # Context-dependent adjustment: W @ brain_features shifts the mean
        self._W = np.zeros((style_dim, brain_dim), dtype=np.float64) * 0.01

        # Running reward baseline for variance reduction
        self._reward_baseline = 0.0
        self._baseline_ema = 0.1

        self._experiences: list[RLExperience] = []
        self._max_history = 1000
        self._total_updates = 0

        self._pre_response_state = None
        self._pre_response_time = 0.0
        self._pending_style = None
        self._pending_context = None
        self._pending_noise = None

        self._reward_history: list[float] = []
        self._style_history: list[dict] = []
        self._param_trajectories = {p: [] for p in STYLE_PARAMS}

    @property
    def current_style(self):
        return self._current_style

    def select_style(self, brain_features):
        """Sample a style vector from the policy given brain context."""
        ctx = np.asarray(brain_features, dtype=np.float64).ravel()
        if len(ctx) < self.brain_dim:
            ctx = np.concatenate([ctx, np.zeros(self.brain_dim - len(ctx))])
        elif len(ctx) > self.brain_dim:
            ctx = ctx[:self.brain_dim]

        # Context-adjusted mean
        mean = self._mean + self._W @ ctx
        std = np.exp(0.5 * self._log_var)
        std = np.clip(std, 0.05, 0.5)

        # Sample from Gaussian policy
        noise = np.random.randn(self.style_dim)
        sampled = mean + std * noise
        sampled = np.clip(sampled, 0, 1)

        self._current_style = StyleVector.from_array(sampled)
        self._pending_style = sampled.copy()
        self._pending_context = ctx.copy()
        self._pending_noise = noise.copy()

        return self._current_style

    def record_pre_response_state(self, brain_state):
        """Call this right before the response is spoken."""
        self._pre_response_state = brain_state
        self._pre_response_time = time.time()

    def compute_reward(self, post_brain_state, jaw_clench=False, error_detected=False):
        """Compute reward from brain state change after response was spoken."""
        if self._pre_response_state is None:
            return 0.0

        pre = self._pre_response_state
        post = post_brain_state

        eng_delta = post.engagement - pre.engagement
        val_shift = post.valence - pre.valence
        no_error = 1.0 - post.error_response
        focus = post.focus

        w = REWARD_WEIGHTS
        reward = (
            w["engagement_delta"] * np.clip(eng_delta * 2, -1, 1)
            + w["valence_shift"] * np.clip(val_shift * 2, -1, 1)
            + w["no_error"] * no_error
            + w["focus"] * focus
        )

        if jaw_clench:
            reward += CLENCH_BONUS
        if error_detected:
            reward += ERRP_PENALTY

        reward = float(np.clip(reward, -1, 1))
        self._update(reward)
        self._pre_response_state = None

        return reward

    def _update(self, reward):
        """REINFORCE policy gradient update."""
        if self._pending_context is None or self._pending_style is None:
            return

        ctx = self._pending_context
        style = self._pending_style
        noise = self._pending_noise

        exp = RLExperience(
            brain_context=ctx.copy(),
            style_used=style.copy(),
            reward=reward,
            timestamp=time.time(),
        )
        if len(self._experiences) < self._max_history:
            self._experiences.append(exp)
        else:
            self._experiences[self._total_updates % self._max_history] = exp

        # Advantage = reward - baseline
        advantage = reward - self._reward_baseline
        self._reward_baseline = (1 - self._baseline_ema) * self._reward_baseline + self._baseline_ema * reward

        # Policy gradient on mean: push mean toward style when advantage > 0
        lr = RL_LEARNING_RATE
        std = np.exp(0.5 * self._log_var)
        std = np.clip(std, 0.05, 0.5)

        # d_log_pi / d_mean = (style - mean) / var = noise / std
        grad_mean = advantage * noise / (std + 1e-8)
        self._mean += lr * grad_mean
        self._mean = np.clip(self._mean, 0.0, 1.0)

        # Also update context weights
        self._W += lr * 0.1 * np.outer(grad_mean, ctx)

        # Update variance: reduce when confident, increase when uncertain
        grad_logvar = advantage * (noise ** 2 - 1) * 0.5
        self._log_var += lr * 0.05 * grad_logvar
        self._log_var = np.clip(self._log_var, -3, 1)

        self._total_updates += 1
        self._reward_history.append(reward)

        style_dict = asdict(StyleVector.from_array(style))
        self._style_history.append(style_dict)
        for i, p in enumerate(STYLE_PARAMS):
            self._param_trajectories[p].append(float(style[i]))

        self._pending_context = None
        self._pending_style = None
        self._pending_noise = None

    def get_stats(self):
        recent_n = min(20, len(self._reward_history))
        recent = self._reward_history[-recent_n:] if recent_n > 0 else []
        return {
            "total_updates": self._total_updates,
            "avg_reward_recent": float(np.mean(recent)) if recent else 0.0,
            "avg_reward_all": float(np.mean(self._reward_history)) if self._reward_history else 0.0,
            "current_style": asdict(self._current_style),
            "exploration_rate": RL_EXPLORATION / (1 + self._total_updates * 0.01),
        }

    def get_trajectories(self):
        return {
            "rewards": self._reward_history,
            "params": self._param_trajectories,
        }

    def save(self, path="neurorl_state.npz"):
        np.savez(path,
                 W=self._W, mean=self._mean, log_var=self._log_var,
                 total_updates=self._total_updates,
                 reward_baseline=self._reward_baseline)

    def load(self, path="neurorl_state.npz"):
        if not Path(path).exists():
            return False
        data = np.load(path)
        self._W = data["W"]
        self._mean = data["mean"]
        self._log_var = data["log_var"]
        self._total_updates = int(data["total_updates"])
        self._reward_baseline = float(data.get("reward_baseline", 0.0))
        return True


class SyntheticUser:
    """Simulated user with known style preferences for RL testing."""

    def __init__(self, preferred_style=None, noise=0.1):
        if preferred_style is None:
            self.preferred = StyleVector(
                formality=0.2, enthusiasm=0.8, verbosity=0.3,
                empathy=0.9, humor=0.6, assertiveness=0.4,
                emotional_expressiveness=0.8, warmth=0.9,
            )
        else:
            self.preferred = preferred_style
        self.noise = noise
        self._pref_arr = self.preferred.to_array()

    def generate_brain_state(self, context="neutral"):
        """Generate synthetic brain features."""
        base = np.array([0.5, 0.4, 0.55, 0.35, 0.45, 0.1,
                        1.2, 1.5, 0.8, 0.7, 0.9, 0.1], dtype=np.float32)

        if context == "engaged":
            base[0] += 0.3
            base[2] += 0.1
            base[5] += 0.2
        elif context == "bored":
            base[0] -= 0.2
            base[4] += 0.3
        elif context == "stressed":
            base[3] += 0.3
            base[2] -= 0.2

        base += np.random.randn(12).astype(np.float32) * 0.05
        return np.clip(base, 0, 1)

    def compute_reward(self, style_vector):
        """Reward based on how close the style is to user's preference."""
        style_arr = style_vector.to_array() if isinstance(style_vector, StyleVector) else style_vector
        distance = np.sqrt(np.mean((style_arr - self._pref_arr) ** 2))
        base_reward = 1.0 - distance
        noise = np.random.randn() * self.noise
        return float(np.clip(base_reward + noise, -1, 1))
