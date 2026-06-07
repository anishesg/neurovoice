#!/usr/bin/env python3
"""NeuroVoice RL Simulation — Demonstrate style learning from synthetic brain signals.

Shows the RL engine converging on a synthetic user's preferred communication style
without any hardware or API calls. The "wow factor" demo.

Usage:
    python simulate.py              # Run 100-round simulation with visualization
    python simulate.py --rounds 200 # Custom number of rounds
    python simulate.py --live       # Run with actual Claude API calls
"""

import sys
import json
import time
import numpy as np
from dataclasses import asdict

from neuro_rl import NeuroRLEngine, StyleVector, SyntheticUser, STYLE_PARAMS
from eeg_pipeline import BrainState


def print_bar(label, value, width=30, color=True):
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {label:<28s} {bar} {value:.3f}")


def print_style_comparison(current, target, title=""):
    if title:
        print(f"\n  {title}")
        print("  " + "─" * 70)
    print(f"  {'Parameter':<28s} {'Current':>8s} {'Target':>8s} {'Gap':>8s}")
    print("  " + "─" * 56)
    for p in STYLE_PARAMS:
        cur = getattr(current, p)
        tgt = getattr(target, p)
        gap = abs(cur - tgt)
        marker = "✓" if gap < 0.1 else "○" if gap < 0.2 else "×"
        print(f"  {p:<28s} {cur:>8.3f} {tgt:>8.3f} {gap:>7.3f} {marker}")


def run_simulation(n_rounds=100, verbose=True):
    """Run RL simulation with synthetic user."""
    user = SyntheticUser(
        preferred_style=StyleVector(
            formality=0.2,
            enthusiasm=0.8,
            verbosity=0.3,
            empathy=0.9,
            humor=0.6,
            assertiveness=0.4,
            emotional_expressiveness=0.8,
            warmth=0.9,
        ),
        noise=0.08,
    )

    rl = NeuroRLEngine()

    contexts = ["neutral", "engaged", "bored", "stressed"]
    conversation_prompts = [
        ("Alice", "How are you doing today?"),
        ("Bob", "Did you see the news about the new AI research?"),
        ("Mom", "Are you eating well?"),
        ("Doctor", "How has your pain been this week?"),
        ("Friend", "Want to hang out this weekend?"),
        ("Colleague", "Can you review this by tomorrow?"),
        ("Stranger", "Excuse me, do you know what time it is?"),
        ("Partner", "I missed you today."),
    ]

    if verbose:
        print("\n" + "=" * 72)
        print("  NEUROVOICE RL SIMULATION")
        print("  Learning communication style from synthetic brain signals")
        print("=" * 72)
        print(f"\n  Target user style (what the RL should converge to):")
        for p in STYLE_PARAMS:
            val = getattr(user.preferred, p)
            print_bar(p, val)

    rewards = []
    style_distances = []

    for round_num in range(1, n_rounds + 1):
        ctx = contexts[round_num % len(contexts)]
        brain_features = user.generate_brain_state(ctx)

        style = rl.select_style(brain_features)

        pre_brain = BrainState(
            engagement=float(brain_features[0]),
            focus=float(brain_features[1]),
            valence=float(brain_features[2]),
            cognitive_load=float(brain_features[3]),
            relaxation=float(brain_features[4]),
        )
        rl.record_pre_response_state(pre_brain)

        base_reward = user.compute_reward(style)

        post_engagement = float(np.clip(brain_features[0] + base_reward * 0.3, 0, 1))
        post_valence = float(np.clip(brain_features[2] + base_reward * 0.2, 0, 1))
        post_brain = BrainState(
            engagement=post_engagement,
            focus=float(brain_features[1]),
            valence=post_valence,
            cognitive_load=float(brain_features[3]),
            relaxation=float(brain_features[4]),
            error_response=0.3 if base_reward < -0.3 else 0.0,
            jaw_clench=base_reward > 0.5 and np.random.random() < 0.3,
        )

        reward = rl.compute_reward(
            post_brain,
            jaw_clench=post_brain.jaw_clench,
            error_detected=post_brain.error_response > 0.5,
        )

        rewards.append(reward)
        dist = np.sqrt(np.mean((style.to_array() - user.preferred.to_array()) ** 2))
        style_distances.append(dist)

        if verbose and (round_num <= 5 or round_num % 10 == 0 or round_num == n_rounds):
            prompt = conversation_prompts[(round_num - 1) % len(conversation_prompts)]
            avg_r = np.mean(rewards[-10:])
            print(f"\n  Round {round_num:3d} | Context: {ctx:<8s} | "
                  f"Reward: {reward:+.3f} | Avg(10): {avg_r:+.3f} | "
                  f"Distance: {dist:.3f}")

            if round_num <= 3 or round_num == n_rounds:
                print(f"  Prompt: \"{prompt[0]}: {prompt[1]}\"")
                print(f"  Style: {style.to_prompt_instructions()[:100]}...")

    if verbose:
        print("\n" + "=" * 72)
        print("  RESULTS")
        print("=" * 72)

        print_style_comparison(rl.current_style, user.preferred,
                              "Final Style vs Target")

        print(f"\n  Performance Summary:")
        print(f"  {'Total rounds:':<28s} {n_rounds}")
        print(f"  {'First 10 avg reward:':<28s} {np.mean(rewards[:10]):+.4f}")
        print(f"  {'Last 10 avg reward:':<28s} {np.mean(rewards[-10:]):+.4f}")
        print(f"  {'Reward improvement:':<28s} {np.mean(rewards[-10:]) - np.mean(rewards[:10]):+.4f}")
        print(f"  {'First 10 avg distance:':<28s} {np.mean(style_distances[:10]):.4f}")
        print(f"  {'Last 10 avg distance:':<28s} {np.mean(style_distances[-10:]):.4f}")
        print(f"  {'Distance improvement:':<28s} {np.mean(style_distances[:10]) - np.mean(style_distances[-10:]):+.4f}")

        converged = sum(1 for p in STYLE_PARAMS
                       if abs(getattr(rl.current_style, p) - getattr(user.preferred, p)) < 0.15)
        print(f"  {'Params within 0.15:':<28s} {converged}/{len(STYLE_PARAMS)}")

        print("\n  Reward trajectory:")
        window = 10
        for i in range(0, len(rewards), max(1, len(rewards) // 10)):
            end = min(i + window, len(rewards))
            avg = np.mean(rewards[i:end])
            bar_val = (avg + 1) / 2
            bar = "█" * int(bar_val * 30)
            print(f"    Round {i+1:>3d}-{end:>3d}: {avg:+.3f} {bar}")

    return {
        "rewards": rewards,
        "style_distances": style_distances,
        "final_style": asdict(rl.current_style),
        "target_style": asdict(user.preferred),
        "trajectories": rl.get_trajectories(),
    }


def run_live_demo():
    """Run with actual Claude API and TTS — requires API keys."""
    from communication_engine import CommunicationEngine
    from voice_output import VoiceOutput
    from style_extractor import extract_imessage_style

    print("\n" + "=" * 72)
    print("  NEUROVOICE LIVE DEMO (Simulation + Claude API + TTS)")
    print("=" * 72)

    style_data = None
    try:
        style_data = extract_imessage_style()
        print(f"  Loaded {style_data['pairs_count']} conversation pairs from iMessages")
    except Exception as e:
        print(f"  iMessage loading skipped: {e}")

    comm = CommunicationEngine(
        style_profile=style_data["profile"] if style_data else None,
        few_shot_examples=style_data["examples"] if style_data else None,
    )
    voice = VoiceOutput()
    rl = NeuroRLEngine()
    user = SyntheticUser()

    conversations = [
        ("Alice", "Hey! How's your day going?"),
        ("Bob", "Did you catch the game last night?"),
        ("Mom", "I made your favorite dinner, when are you coming over?"),
        ("Doctor", "Your test results came back, everything looks normal."),
        ("Friend", "We're all going to that new restaurant Saturday, you in?"),
    ]

    for speaker, message in conversations:
        print(f"\n{'─' * 72}")
        print(f"  {speaker}: \"{message}\"")

        brain_features = user.generate_brain_state("engaged")
        style = rl.select_style(brain_features)

        brain = BrainState(
            engagement=float(brain_features[0]),
            focus=float(brain_features[1]),
            valence=float(brain_features[2]),
            cognitive_load=float(brain_features[3]),
            relaxation=float(brain_features[4]),
        )

        comm.add_incoming_message(speaker, message)
        intent, conf = comm.classify_intent(brain)
        print(f"  Brain intent: {intent} ({conf:.0%})")
        print(f"  Style: {style.to_prompt_instructions()[:80]}...")

        print(f"\n  Generating responses...")
        candidates = comm.generate_responses(brain, style)

        for c in candidates:
            print(f"    [{c.index + 1}] {c.text}")

        if candidates:
            selected = candidates[0]
            print(f"\n  Selected: \"{selected.text}\"")
            comm.record_selected_response(selected)

            rl.record_pre_response_state(brain)
            reward = user.compute_reward(style)
            post_brain = BrainState(
                engagement=float(np.clip(brain_features[0] + reward * 0.2, 0, 1)),
                valence=float(np.clip(brain_features[2] + reward * 0.15, 0, 1)),
            )
            actual_reward = rl.compute_reward(post_brain)
            print(f"  Neural reward: {actual_reward:+.3f}")

            print(f"  Speaking...")
            try:
                voice.speak(selected.text, brain, blocking=True)
                print(f"  Spoken successfully")
            except Exception as e:
                print(f"  Voice error: {e}")

    print(f"\n{'=' * 72}")
    print(f"  Demo complete. RL stats: {json.dumps(rl.get_stats(), indent=2)}")


if __name__ == "__main__":
    n_rounds = 100
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--rounds" and i < len(sys.argv) - 1:
            n_rounds = int(sys.argv[i + 1])
        elif arg == "--live":
            run_live_demo()
            sys.exit(0)

    results = run_simulation(n_rounds=n_rounds, verbose=True)
