#!/usr/bin/env python3
"""NeuroVoice Server — WebSocket orchestration of EEG + RL + Claude + TTS.

Usage:
    python server.py              # Real Muse S EEG
    python server.py --simulate   # Synthetic EEG (no headband)
"""

import asyncio
import json
import sys
import time
import os

import numpy as np
import websockets

from config import WS_HOST, WS_PORT, EEG_SAMPLE_RATE, NUM_CANDIDATES
from eeg_pipeline import EEGPipeline, BrainState
from neuro_rl import NeuroRLEngine, StyleVector
from communication_engine import CommunicationEngine, ResponseCandidate
from voice_output import VoiceOutput
from style_extractor import extract_imessage_style

SIM_MODE = "--simulate" in sys.argv or not os.environ.get("MUSE_CONNECTED")


class NeuroVoiceServer:
    def __init__(self, simulate=False):
        self.eeg = EEGPipeline(simulate=simulate)
        self.rl = NeuroRLEngine()
        self.voice = VoiceOutput()
        self.comm = None
        self.clients: set = set()

        self.phase = "startup"
        self.current_candidates: list[ResponseCandidate] = []
        self.selected_index = -1
        self.latest_brain = BrainState()
        self.latest_intent = ("acknowledge", 0.5)
        self.latest_style = StyleVector()

        self._candidate_engagement = {}
        self._candidate_dwell_start = {}
        self._focused_candidate = -1
        self._selection_threshold = 2.0
        self._total_interactions = 0

    async def initialize(self):
        self.phase = "loading_style"
        await self._broadcast("system", {"status": "Loading communication style..."})

        style_data = None
        try:
            style_data = extract_imessage_style()
            profile = style_data["profile"]
            examples = style_data["examples"]
            await self._broadcast("system", {
                "status": f"Loaded style from {style_data['pairs_count']} iMessage conversations",
            })
        except Exception as e:
            profile = None
            examples = None
            await self._broadcast("system", {"status": f"Style loading skipped: {e}"})

        self.comm = CommunicationEngine(
            style_profile=profile,
            few_shot_examples=examples,
        )

        self.phase = "connecting_eeg"
        await self._broadcast("system", {"status": "Connecting EEG..."})
        try:
            self.eeg.connect()
            await self._broadcast("system", {"status": "EEG connected"})
        except Exception as e:
            await self._broadcast("system", {"status": f"EEG error: {e}"})

        if os.path.exists("neurorl_state.npz"):
            self.rl.load("neurorl_state.npz")
            await self._broadcast("system", {"status": "Loaded RL state from disk"})

        self.phase = "ready"
        await self._broadcast("phase", {"phase": "ready"})

    async def eeg_loop(self):
        while True:
            self.eeg.poll()
            await asyncio.sleep(0.04)

    async def brain_loop(self):
        while True:
            self.latest_brain = self.eeg.update_brain_state()

            brain_data = {
                "engagement": round(self.latest_brain.engagement, 4),
                "focus": round(self.latest_brain.focus, 4),
                "valence": round(self.latest_brain.valence, 4),
                "cognitive_load": round(self.latest_brain.cognitive_load, 4),
                "relaxation": round(self.latest_brain.relaxation, 4),
                "asymmetry": round(self.latest_brain.asymmetry, 4),
                "jaw_clench": self.latest_brain.jaw_clench,
                "error_response": round(self.latest_brain.error_response, 4),
                "band_powers": {k: round(v, 2) for k, v in self.latest_brain.band_powers.items()},
            }
            await self._broadcast("brain", brain_data)

            if self.comm:
                intent, conf = self.comm.classify_intent(self.latest_brain)
                self.latest_intent = (intent, conf)
                await self._broadcast("intent", {"intent": intent, "confidence": round(conf, 3)})

            if self.latest_brain.jaw_clench and self._focused_candidate >= 0:
                await self._select_candidate(self._focused_candidate)

            await asyncio.sleep(0.1)

    async def handle_ws(self, ws):
        self.clients.add(ws)
        await ws.send(json.dumps({
            "type": "init",
            "phase": self.phase,
            "sim_mode": self.eeg.simulate,
            "rl_stats": self.rl.get_stats(),
            "candidates": [{"text": c.text, "intent": c.intent, "index": c.index}
                          for c in self.current_candidates],
        }))
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    await self._handle_command(data)
                except json.JSONDecodeError:
                    pass
        finally:
            self.clients.discard(ws)

    async def _handle_command(self, data):
        cmd = data.get("command")

        if cmd == "incoming_message":
            speaker = data.get("speaker", "Someone")
            text = data.get("text", "")
            if text and self.comm:
                self.comm.add_incoming_message(speaker, text)
                await self._broadcast("conversation", {
                    "speaker": speaker, "text": text,
                })
                await self._generate_responses()

        elif cmd == "generate":
            await self._generate_responses()

        elif cmd == "select":
            idx = data.get("index", 0)
            await self._select_candidate(idx)

        elif cmd == "focus_candidate":
            self._focused_candidate = data.get("index", -1)

        elif cmd == "set_sim_bias":
            self.eeg.set_sim_bias(
                data.get("engagement", 0.5),
                data.get("valence", 0.5),
            )

        elif cmd == "save_rl":
            self.rl.save("neurorl_state.npz")
            await self._broadcast("system", {"status": "RL state saved"})

    async def _generate_responses(self):
        if not self.comm:
            return

        self.phase = "generating"
        await self._broadcast("phase", {"phase": "generating"})

        brain_features = self.latest_brain.to_feature_vector()
        self.latest_style = self.rl.select_style(brain_features)

        try:
            self.current_candidates = self.comm.generate_responses(
                self.latest_brain, self.latest_style,
            )
        except Exception as e:
            print(f"[Server] Generation error: {e}")
            self.current_candidates = []

        self._candidate_engagement = {}
        self._candidate_dwell_start = {}
        self._focused_candidate = -1

        self.rl.record_pre_response_state(self.latest_brain)

        self.phase = "selecting"
        await self._broadcast("candidates", {
            "candidates": [{"text": c.text, "intent": c.intent, "index": c.index}
                          for c in self.current_candidates],
            "style": self.latest_style.to_prompt_instructions(),
            "intent": self.latest_intent[0],
            "intent_confidence": round(self.latest_intent[1], 3),
        })

    async def _select_candidate(self, index):
        if index < 0 or index >= len(self.current_candidates):
            return

        candidate = self.current_candidates[index]
        self.selected_index = index

        self.comm.record_selected_response(candidate)

        await self._broadcast("selected", {
            "index": index,
            "text": candidate.text,
            "intent": candidate.intent,
        })

        self.phase = "speaking"
        await self._broadcast("phase", {"phase": "speaking"})

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, self.voice.speak, candidate.text, self.latest_brain, True,
            )
        except Exception as e:
            print(f"[Server] Voice error: {e}")

        await asyncio.sleep(1.0)
        post_brain = self.eeg.update_brain_state()
        reward = self.rl.compute_reward(
            post_brain,
            jaw_clench=post_brain.jaw_clench,
            error_detected=post_brain.error_response > 0.5,
        )

        self._total_interactions += 1

        await self._broadcast("reward", {
            "reward": round(reward, 4),
            "rl_stats": self.rl.get_stats(),
            "interaction": self._total_interactions,
        })

        self.phase = "ready"
        await self._broadcast("phase", {"phase": "ready"})
        await self._broadcast("conversation_update", {
            "context": self.comm.get_conversation_context(),
        })

    async def _broadcast(self, msg_type, data):
        msg = json.dumps({"type": msg_type, "timestamp": time.time(), **data})
        dead = set()
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self.clients -= dead


async def main():
    server = NeuroVoiceServer(simulate=SIM_MODE)

    ws_server = await websockets.serve(server.handle_ws, WS_HOST, WS_PORT)
    print(f"\nNeuroVoice server on ws://{WS_HOST}:{WS_PORT}")
    print(f"  Mode: {'SIMULATION' if SIM_MODE else 'LIVE (Muse S)'}")

    await server.initialize()

    eeg_task = asyncio.create_task(server.eeg_loop())
    brain_task = asyncio.create_task(server.brain_loop())

    print(f"\nReady. Open frontend/index.html in a browser.")
    print(f"Or send messages via WebSocket: {{\"command\": \"incoming_message\", \"speaker\": \"Alice\", \"text\": \"How are you?\"}}\n")

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        eeg_task.cancel()
        brain_task.cancel()
        server.eeg.stop()
        server.rl.save("neurorl_state.npz")
        ws_server.close()


if __name__ == "__main__":
    asyncio.run(main())
