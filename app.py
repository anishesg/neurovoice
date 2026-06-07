#!/usr/bin/env python3
"""NeuroVoice — Real-time EEG + ElevenLabs Conversational AI.

ElevenLabs handles STT (Scribe) + TTS (v3 with emotion).
Bedrock Claude generates brain-state-aware responses.
Muse S EEG provides brain state via muse_bridge shared memory.

Usage:
    python app.py              # live Muse S
    python app.py --simulate   # synthetic EEG
"""

import asyncio
import base64
import json
import os
import sys
import struct
import time
import threading
import io
import wave

import numpy as np
import websockets
import pyaudio
from multiprocessing import shared_memory, resource_tracker
from elevenlabs.client import ElevenLabs as EL
from elevenlabs import VoiceSettings

from config import WS_HOST, WS_PORT, EEG_SAMPLE_RATE as SR
from eeg_pipeline import BrainStateEngine, BrainState

sys.path.insert(0, os.path.expanduser("~/axiom"))
from neuralrl import BrainReward, CTS, NeuralFingerprint, AdaptationTracker
from neuralrl import BrainPublisher, BrainInterpreter, WeaveTracker
from neuralrl.brain_reward import BrainSnapshot

STYLE_PARAMS = ["formality", "enthusiasm", "verbosity", "empathy",
                "humor", "assertiveness", "expressiveness", "warmth"]

SIM_MODE = "--simulate" in sys.argv
ELEVEN_KEY = os.environ.get("ELEVEN_API_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George - natural male
TTS_MODEL = "eleven_v3"

SHM_DATA_NAME = "muse_eeg_ring"
SHM_META_NAME = "muse_eeg_meta"
META_FORMAT = "<qqbfd i"
META_SIZE = struct.calcsize(META_FORMAT)
N_CH = 4
RING_SAMPLES = SR * 10

MIC_RATE = 16000
MIC_CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_GAP = 1.2
MIN_SPEECH = 0.4


# ── EEG from muse_bridge shared memory ─────────────────────────
class LiveEEG:
    def __init__(self, simulate=False):
        self.simulate = simulate
        self._shm_data = None
        self._shm_meta = None
        self._ring = None
        self._local_ring = np.zeros((N_CH, SR * 10), dtype=np.float64)
        self._local_pos = 0
        self._last_read_total = 0
        self._sim_phase = 0.0
        self._sim_bias = {"eng": 0.5, "val": 0.5}
        self.connected = False

    def connect(self):
        if self.simulate:
            self.connected = True
            return True
        try:
            self._shm_data = shared_memory.SharedMemory(name=SHM_DATA_NAME, create=False)
            resource_tracker.unregister(self._shm_data._name, "shared_memory")
            self._shm_meta = shared_memory.SharedMemory(name=SHM_META_NAME, create=False)
            resource_tracker.unregister(self._shm_meta._name, "shared_memory")
            self._ring = np.ndarray((N_CH, RING_SAMPLES), dtype=np.float64, buffer=self._shm_data.buf)
            meta = self._read_meta()
            self._last_read_total = meta["total"]
            self.connected = meta["connected"]
            print(f"[EEG] Connected to bridge (pid={meta['pid']})", flush=True)
            return True
        except FileNotFoundError:
            print("[EEG] Bridge not found, falling back to sim", flush=True)
            self.simulate = True
            self.connected = True
            return True

    def _read_meta(self):
        if not self._shm_meta:
            return {"write_pos": 0, "total": 0, "connected": False, "quality": 0, "timestamp": 0, "pid": 0}
        data = bytes(self._shm_meta.buf[:META_SIZE])
        wp, total, connected, quality, ts, pid = struct.unpack(META_FORMAT, data)
        return {"write_pos": wp, "total": total, "connected": bool(connected), "quality": quality, "timestamp": ts, "pid": pid}

    def poll(self):
        if self.simulate:
            self._gen_sim()
            return
        meta = self._read_meta()
        self.connected = meta["connected"]
        new = meta["total"] - self._last_read_total
        if new <= 0:
            return
        new = min(new, RING_SAMPLES)
        end = meta["write_pos"]
        start = (end - new) % RING_SAMPLES
        data = self._ring[:, start:end].copy() if start < end else np.concatenate([self._ring[:, start:], self._ring[:, :end]], axis=1)
        self._last_read_total = meta["total"]
        self._write_local(data)

    def get_window(self, seconds=1.0):
        n = min(int(seconds * SR), self._local_pos, self._local_ring.shape[1])
        if n <= 0:
            return np.zeros((4, 0))
        cap = self._local_ring.shape[1]
        end = self._local_pos % cap
        if n <= end:
            return self._local_ring[:, end - n:end].copy()
        return np.concatenate([self._local_ring[:, cap - (n - end):], self._local_ring[:, :end]], axis=1)

    def _write_local(self, data):
        cap = self._local_ring.shape[1]
        for i in range(data.shape[1]):
            self._local_ring[:, self._local_pos % cap] = data[:, i]
            self._local_pos += 1

    def set_sim_bias(self, eng=0.5, val=0.5):
        self._sim_bias = {"eng": eng, "val": val}

    def _gen_sim(self):
        n = 32
        t = np.arange(n) / SR + self._sim_phase
        self._sim_phase += n / SR
        eng, val = self._sim_bias["eng"], self._sim_bias["val"]
        eeg = np.zeros((4, n))
        for ch in range(4):
            p = ch * 0.5
            eeg[ch] += (15 * (1 - eng * 0.5)) * np.sin(2 * np.pi * 10 * t + p)
            eeg[ch] += (8 * (0.3 + eng * 0.7)) * np.sin(2 * np.pi * 22 * t + p * 0.7)
            eeg[ch] += (10 * (0.5 + (1 - eng) * 0.3)) * np.sin(2 * np.pi * 6 * t + p * 1.2)
            eeg[ch] += np.random.randn(n) * 4
        eeg[1] += (val - 0.5) * 6 * np.sin(2 * np.pi * 10 * t)
        eeg[2] -= (val - 0.5) * 6 * np.sin(2 * np.pi * 10 * t)
        self._write_local(eeg)

    def stop(self):
        for shm in [self._shm_data, self._shm_meta]:
            if shm:
                try:
                    shm.close()
                except:
                    pass


# ── Intent classifier ──────────────────────────────────────────
class IntentClassifier:
    def classify(self, b):
        if b.error_response > 0.5:
            return "correct", min(1.0, b.error_response)
        if b.engagement > 0.65 and b.valence > 0.6:
            return "agree", min(0.9, b.engagement)
        if b.engagement > 0.65 and b.valence < 0.4:
            return "disagree", min(0.9, b.engagement)
        if b.cognitive_load > 0.6 and b.focus > 0.5:
            return "elaborate", 0.8
        if b.engagement > 0.5 and b.cognitive_load > 0.4:
            return "question", 0.6
        if b.relaxation > 0.6 and b.valence > 0.6:
            return "express_emotion", 0.7
        if b.engagement < 0.3:
            return "acknowledge", 0.5
        return "acknowledge", 0.5


# ── Emotion → ElevenLabs v3 audio tags ─────────────────────────
def brain_to_emotion_tag(brain):
    val, eng, relax = brain.valence, brain.engagement, brain.relaxation
    arousal = eng * 0.6 + (1 - relax) * 0.4
    if val > 0.65 and arousal > 0.6:
        return "[excited]"
    if val > 0.65:
        return "[happy]"
    if val < 0.35 and arousal > 0.6:
        return "[angry]"
    if val < 0.35:
        return "[sad]"
    if eng > 0.7:
        return ""
    if eng < 0.25:
        return "[tired]"
    return ""


# ── Mic capture + ElevenLabs STT (Scribe) ──────────────────────
class MicSTT:
    """Captures mic audio, detects speech, transcribes via ElevenLabs Scribe."""

    def __init__(self):
        self._el = EL(api_key=ELEVEN_KEY)
        self._running = False
        self._on_hearing = None
        self._on_partial = None
        self._on_final = None
        self._audio_buffer = []
        self._is_speaking = False
        self._last_speech = 0.0
        self._speech_start = 0.0
        self.muted = False

    def start(self, on_hearing=None, on_partial=None, on_final=None):
        self._on_hearing = on_hearing
        self._on_partial = on_partial
        self._on_final = on_final
        self._running = True
        threading.Thread(target=self._capture, daemon=True).start()
        print("[STT] ElevenLabs Scribe mic capture started", flush=True)

    def stop(self):
        self._running = False

    def _capture(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=MIC_RATE,
                        input=True, frames_per_buffer=MIC_CHUNK)
        try:
            while self._running:
                data = stream.read(MIC_CHUNK, exception_on_overflow=False)

                if self.muted:
                    if self._is_speaking:
                        self._is_speaking = False
                        self._audio_buffer = []
                        if self._on_hearing:
                            self._on_hearing(False)
                    continue

                chunk = np.frombuffer(data, dtype=np.int16)
                energy = float(np.mean(np.abs(chunk)))
                now = time.time()

                if energy > SILENCE_THRESHOLD:
                    if not self._is_speaking:
                        self._is_speaking = True
                        self._speech_start = now
                        self._audio_buffer = []
                        if self._on_hearing:
                            self._on_hearing(True)
                    self._last_speech = now
                    self._audio_buffer.append(data)
                elif self._is_speaking:
                    self._audio_buffer.append(data)
                    if now - self._last_speech >= SILENCE_GAP:
                        dur = self._last_speech - self._speech_start
                        if dur >= MIN_SPEECH and self._audio_buffer:
                            self._transcribe()
                        self._is_speaking = False
                        self._audio_buffer = []
                        if self._on_hearing:
                            self._on_hearing(False)

                if self._is_speaking and self._on_partial:
                    dur = now - self._speech_start
                    if dur > 1.0:
                        self._on_partial(f"(listening... {dur:.0f}s)")

        finally:
            stream.close()
            pa.terminate()

    def _transcribe(self):
        raw = b"".join(self._audio_buffer)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(MIC_RATE)
            wf.writeframes(raw)
        buf.seek(0)
        buf.name = "speech.wav"

        try:
            result = self._el.speech_to_text.convert(
                file=buf, model_id="scribe_v2", language_code="en",
            )
            text = result.text.strip()
            if text:
                print(f'[STT] "{text}"', flush=True)
                if self._on_final:
                    self._on_final(text)
        except Exception as e:
            print(f"[STT] Error: {e}", flush=True)


# ── ElevenLabs TTS ─────────────────────────────────────────────
class ElevenTTS:
    def __init__(self):
        self._el = EL(api_key=ELEVEN_KEY)

    def speak(self, text, brain=None):
        """Stream speech — start playing as soon as first chunks arrive."""
        tag = brain_to_emotion_tag(brain) if brain else ""
        tagged_text = f"{tag} {text}" if tag else text

        try:
            from elevenlabs import stream as el_stream
            audio_stream = self._el.text_to_speech.stream(
                text=tagged_text,
                voice_id=VOICE_ID,
                model_id=TTS_MODEL,
                output_format="mp3_44100_128",
                voice_settings=VoiceSettings(
                    stability=0.35,
                    similarity_boost=0.85,
                    style=0.2,
                    speed=1.05,
                ),
            )
            el_stream(audio_stream)
        except Exception as e:
            print(f"[TTS] Error: {e}", flush=True)


# ── Claude response generation via Bedrock ─────────────────────
class BrainLLM:
    """OpenAI-powered brain-to-speech with Structured Outputs."""

    def __init__(self):
        self._interpreter = BrainInterpreter(api_key=OPENAI_KEY, model="gpt-4o-mini")
        self._conversation = []
        self._max_ctx = 20

    def add_message(self, speaker, text):
        self._conversation.append({"speaker": speaker, "text": text})
        if len(self._conversation) > self._max_ctx:
            self._conversation = self._conversation[-self._max_ctx:]

    def generate(self, brain, style):
        intent_clf = IntentClassifier()
        intent, conf = intent_clf.classify(brain)

        ctx = "\n".join(f"{'Me' if m['speaker'] == 'me' else m['speaker']}: {m['text']}"
                       for m in self._conversation[-8:])

        brain_features = {
            "engagement": brain.engagement, "focus": brain.focus,
            "valence": brain.valence, "cognitive_load": brain.cognitive_load,
            "relaxation": brain.relaxation, "asymmetry": brain.asymmetry,
            "intent": intent, "intent_confidence": conf,
        }

        style_params = {p: getattr(style, p, 0.5) for p in
            ["formality","enthusiasm","verbosity","empathy","humor",
             "assertiveness","emotional_expressiveness","warmth"]}

        try:
            result = self._interpreter.interpret(brain_features, ctx, style_params)
            candidates = [result.phrase] + result.alternatives[:3]
            print(f"[LLM] intent={result.intent} emotion={result.emotion} conf={result.confidence:.2f}", flush=True)
            return candidates, intent, conf
        except Exception as e:
            print(f"[LLM] Error: {e}", flush=True)
            return ["Yeah, I hear you.", "For sure."], intent, conf


# ── Main App ───────────────────────────────────────────────────
class NeuroVoiceApp:
    def __init__(self):
        self.eeg = LiveEEG(simulate=SIM_MODE)
        self.brain_engine = BrainStateEngine()
        self.reward = BrainReward()
        self.policy = CTS(action_dim=8, context_dim=12, param_names=STYLE_PARAMS)
        self.fingerprint = NeuralFingerprint()
        self.adaptation = AdaptationTracker()
        self.llm = BrainLLM()
        self.stt = MicSTT()
        self.tts = ElevenTTS()
        self.redis = BrainPublisher()
        if self.redis.connect():
            print("[REDIS] Brain stream publishing enabled", flush=True)
        self.weave = WeaveTracker()
        self.weave.init()
        self.clients = set()

        self.brain = BrainState()
        self.brain_snap = BrainSnapshot()
        self.brain_history = []
        self.conversation = []
        self.candidates = []
        self.phase = "startup"
        self._tick = 0
        self._generating = False
        self._speaking = False
        self._loop = None

    async def start(self):
        self._loop = asyncio.get_event_loop()
        self.eeg.connect()
        self.stt.start(
            on_hearing=self._on_hearing,
            on_partial=self._on_partial,
            on_final=self._on_final,
        )
        self.phase = "live"
        await self._broadcast("phase", {"phase": "live"})

    def _on_hearing(self, active):
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast("stt", {"hearing": active, "text": "", "final": False}),
                self._loop)

    def _on_partial(self, text):
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast("stt", {"hearing": True, "text": text, "final": False}),
                self._loop)

    def _on_final(self, text):
        if not text or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._handle_speech(text), self._loop)

    async def _handle_speech(self, text):
        await self._broadcast("stt", {"text": text, "final": True, "hearing": False})
        self.llm.add_message("Them", text)
        self.conversation.append({"speaker": "Them", "text": text})
        await self._broadcast("conversation", {"conversation": self.conversation})
        if not self._generating:
            asyncio.create_task(self._generate_and_speak())

    async def _generate_and_speak(self):
        self._generating = True
        self.phase = "generating"
        await self._broadcast("phase", {"phase": "generating"})

        ctx = self.reward.get_context(self.brain_snap)
        action = self.policy.sample(ctx)
        style_dict = {p: float(action[i]) for i, p in enumerate(STYLE_PARAMS)}

        from neuro_rl import StyleVector
        style = StyleVector(**{p: style_dict.get(p, 0.5) for p in
            ["formality","enthusiasm","verbosity","empathy","humor",
             "assertiveness","emotional_expressiveness","warmth"]})

        loop = asyncio.get_event_loop()
        try:
            candidates, intent, conf = await loop.run_in_executor(
                None, self.llm.generate, self.brain, style)
            self.candidates = candidates[:2]
        except Exception as e:
            print(f"[APP] Gen error: {e}", flush=True)
            self.candidates = ["Yeah, I hear you.", "For sure."]
            intent, conf = "acknowledge", 0.5

        if len(self.candidates) < 2:
            self.candidates.append("Yeah, I hear you.")

        # ── PHASE 1: Show response A, measure brain ──
        self.phase = "evaluating"
        await self._broadcast("eval_start", {
            "intent": intent, "intent_conf": round(conf, 3),
            "style": style_dict, "posteriors": self.policy.get_posteriors(),
        })

        # Response A
        await self._broadcast("eval_response", {
            "index": 0, "text": self.candidates[0], "label": "A",
        })
        brain_samples_a = []
        reading_time = max(2.0, len(self.candidates[0].split()) * 0.25)
        t0 = time.time()
        while time.time() - t0 < reading_time:
            brain_samples_a.append({
                "engagement": self.brain.engagement,
                "valence": self.brain.valence,
                "focus": self.brain.focus,
            })
            await self._broadcast("eval_brain_sample", {
                "index": 0, "label": "A",
                "engagement": round(self.brain.engagement, 4),
                "valence": round(self.brain.valence, 4),
                "focus": round(self.brain.focus, 4),
                "t": round(time.time() - t0, 2),
            })
            await asyncio.sleep(0.2)

        avg_a_eng = np.mean([s["engagement"] for s in brain_samples_a]) if brain_samples_a else 0
        avg_a_val = np.mean([s["valence"] for s in brain_samples_a]) if brain_samples_a else 0.5
        score_a = float(avg_a_eng * 0.6 + avg_a_val * 0.4)

        await self._broadcast("eval_score", {
            "index": 0, "label": "A", "score": round(score_a, 4),
            "avg_engagement": round(float(avg_a_eng), 4),
            "avg_valence": round(float(avg_a_val), 4),
        })

        await asyncio.sleep(0.5)

        # ── PHASE 2: Show response B, measure brain ──
        await self._broadcast("eval_response", {
            "index": 1, "text": self.candidates[1], "label": "B",
        })
        brain_samples_b = []
        reading_time = max(2.0, len(self.candidates[1].split()) * 0.25)
        t0 = time.time()
        while time.time() - t0 < reading_time:
            brain_samples_b.append({
                "engagement": self.brain.engagement,
                "valence": self.brain.valence,
                "focus": self.brain.focus,
            })
            await self._broadcast("eval_brain_sample", {
                "index": 1, "label": "B",
                "engagement": round(self.brain.engagement, 4),
                "valence": round(self.brain.valence, 4),
                "focus": round(self.brain.focus, 4),
                "t": round(time.time() - t0, 2),
            })
            await asyncio.sleep(0.2)

        avg_b_eng = np.mean([s["engagement"] for s in brain_samples_b]) if brain_samples_b else 0
        avg_b_val = np.mean([s["valence"] for s in brain_samples_b]) if brain_samples_b else 0.5
        score_b = float(avg_b_eng * 0.6 + avg_b_val * 0.4)

        await self._broadcast("eval_score", {
            "index": 1, "label": "B", "score": round(score_b, 4),
            "avg_engagement": round(float(avg_b_eng), 4),
            "avg_valence": round(float(avg_b_val), 4),
        })

        # ── PHASE 3: Pick winner, speak it ──
        winner = 0 if score_a >= score_b else 1
        selected = self.candidates[winner]

        self.reward.mark_pre_action(self.brain_snap)

        await self._broadcast("eval_winner", {
            "winner": winner,
            "winner_label": "A" if winner == 0 else "B",
            "text": selected,
            "score_a": round(score_a, 4),
            "score_b": round(score_b, 4),
            "margin": round(abs(score_a - score_b), 4),
        })

        await asyncio.sleep(0.8)

        self.llm.add_message("me", selected)
        self.conversation.append({"speaker": "Me", "text": selected})
        await self._broadcast("conversation", {"conversation": self.conversation})

        self.phase = "speaking"
        await self._broadcast("phase", {"phase": "speaking"})

        self.stt.muted = True
        try:
            await loop.run_in_executor(None, self.tts.speak, selected, self.brain)
        except Exception as e:
            print(f"[APP] TTS error: {e}", flush=True)
        finally:
            await asyncio.sleep(0.3)
            self.stt.muted = False

        post_snap = BrainSnapshot(
            engagement=self.brain.engagement, focus=self.brain.focus,
            valence=self.brain.valence, cognitive_load=self.brain.cognitive_load,
            relaxation=self.brain.relaxation, error_response=self.brain.error_response,
            jaw_clench=self.brain.jaw_clench, band_powers=self.brain.band_powers,
            asymmetry=self.brain.asymmetry, timestamp=time.time(),
        )
        reward = self.reward.compute(post_snap)
        self.policy.update(reward)

        adapt_score = self.adaptation.update(
            reward, self.policy.exploration_rate, self.policy.std)

        # ── Weave: trace + log + publish ──
        winner_label = "A" if winner == 0 else "B"
        brain_features = {
            "engagement": self.brain.engagement, "focus": self.brain.focus,
            "valence": self.brain.valence, "cognitive_load": self.brain.cognitive_load,
        }
        self.weave.trace_pipeline(
            brain_features, intent, style_dict, self.candidates,
            score_a, score_b, winner_label,
            brain_samples_a, brain_samples_b,
            reward, adapt_score, self.policy._total_updates)
        self.weave.log_episode(
            ctx.tolist(), style_dict, self.candidates,
            score_a, score_b, winner_label, reward, adapt_score,
            self.policy._total_updates, intent)
        if self.policy._total_updates % 5 == 0:
            self.weave.publish_policy(self.policy, STYLE_PARAMS)
        if self.policy._total_updates % 20 == 0:
            self.weave.run_eval(self.policy, STYLE_PARAMS)
        self.redis.publish_rl_update(reward, self.policy.get_posteriors(), adapt_score)

        self._generating = False
        self.phase = "live"
        await self._broadcast("reward", {
            "reward": round(reward, 4),
            "adaptation": round(adapt_score, 1),
            "adaptation_breakdown": self.adaptation.get_breakdown(),
            "posteriors": self.policy.get_posteriors(),
            "trajectories": self.policy.get_trajectories(),
            "reward_history": self.reward.get_history(),
            "rl_stats": self.policy.get_stats(),
            "weave_episodes": self.weave.episode_count,
        })
        await self._broadcast("phase", {"phase": "live"})

    async def eeg_loop(self):
        while True:
            self.eeg.poll()
            await asyncio.sleep(0.04)

    async def brain_loop(self):
        while True:
            window = self.eeg.get_window(1.0)
            if window.shape[1] >= SR:
                self.brain = self.brain_engine.process(window, timestamp=time.time())
                self._tick += 1
                intent_clf = IntentClassifier()
                intent, conf = intent_clf.classify(self.brain)

                snap = {
                    "t": self._tick,
                    "engagement": round(self.brain.engagement, 4),
                    "focus": round(self.brain.focus, 4),
                    "valence": round(self.brain.valence, 4),
                    "cognitive_load": round(self.brain.cognitive_load, 4),
                    "relaxation": round(self.brain.relaxation, 4),
                    "asymmetry": round(self.brain.asymmetry, 4),
                    "jaw_clench": self.brain.jaw_clench,
                    "error_response": round(self.brain.error_response, 4),
                    "intent": intent,
                    "intent_conf": round(conf, 3),
                    "bands": {k: round(v, 2) for k, v in self.brain.band_powers.items()},
                    "connected": self.eeg.connected,
                }
                self.brain_history.append(snap)
                if len(self.brain_history) > 600:
                    self.brain_history = self.brain_history[-600:]

                # Update neuralrl components
                self.brain_snap = BrainSnapshot(
                    engagement=self.brain.engagement, focus=self.brain.focus,
                    valence=self.brain.valence, cognitive_load=self.brain.cognitive_load,
                    relaxation=self.brain.relaxation, error_response=self.brain.error_response,
                    jaw_clench=self.brain.jaw_clench, band_powers=self.brain.band_powers,
                    asymmetry=self.brain.asymmetry, timestamp=time.time(),
                )
                fp = self.fingerprint.update(self.brain_snap)
                self.redis.publish_brain(self.brain_snap)

                await self._broadcast("tick", {
                    "brain": snap,
                    "fingerprint": fp,
                    "posteriors": self.policy.get_posteriors(),
                    "adaptation": round(self.adaptation.score, 1),
                    "adaptation_breakdown": self.adaptation.get_breakdown(),
                    "rl_stats": self.policy.get_stats(),
                })
            await asyncio.sleep(0.1)

    async def handle_ws(self, ws):
        self.clients.add(ws)
        await ws.send(json.dumps({
            "type": "init",
            "phase": self.phase,
            "conversation": self.conversation,
            "candidates": [{"text": c, "index": i} for i, c in enumerate(self.candidates)],
            "rl_stats": self.policy.get_stats(),
        }))
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    cmd = data.get("cmd")
                    if cmd == "new_chat":
                        self.conversation = []
                        self.candidates = []
                        self.llm._conversation = []
                        self._generating = False
                        await self._broadcast("conversation", {"conversation": []})
                        await self._broadcast("candidates", {"candidates": []})
                    elif cmd == "message":
                        text = data.get("text", "").strip()
                        if text:
                            await self._handle_speech(text)
                    elif cmd == "select":
                        idx = data.get("index", 0)
                        if 0 <= idx < len(self.candidates):
                            sel = self.candidates[idx]
                            self.llm.add_message("me", sel)
                            self.conversation.append({"speaker": "Me", "text": sel})
                            await self._broadcast("selected", {"index": idx, "text": sel})
                            await self._broadcast("conversation", {"conversation": self.conversation})
                            loop = asyncio.get_event_loop()
                            asyncio.create_task(self._speak_muted(sel))
                except json.JSONDecodeError:
                    pass
        finally:
            self.clients.discard(ws)

    async def _speak_muted(self, text):
        self.stt.muted = True
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.tts.speak, text, self.brain)
        finally:
            await asyncio.sleep(0.3)
            self.stt.muted = False

    async def _broadcast(self, msg_type, data):
        msg = json.dumps({"type": msg_type, **data})
        dead = set()
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except:
                dead.add(ws)
        self.clients -= dead


async def main():
    app = NeuroVoiceApp()
    ws_server = await websockets.serve(app.handle_ws, WS_HOST, WS_PORT)
    print(f"\n  NeuroVoice on ws://{WS_HOST}:{WS_PORT}", flush=True)
    print(f"  Mode: {'SIM' if SIM_MODE else 'LIVE Muse S'}", flush=True)
    print(f"  STT: ElevenLabs Scribe", flush=True)
    print(f"  TTS: ElevenLabs v3 ({VOICE_ID})", flush=True)
    print(f"  LLM: OpenAI gpt-4o-mini (Structured Outputs)\n", flush=True)

    await app.start()
    await asyncio.gather(
        app.eeg_loop(),
        app.brain_loop(),
        asyncio.ensure_future(asyncio.sleep(float("inf"))),
    )

if __name__ == "__main__":
    asyncio.run(main())
