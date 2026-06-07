"""Voice Output — Emotion-aware text-to-speech.

Maps EEG brain state to emotional voice parameters and synthesizes speech
using OpenAI gpt-4o-mini-tts with natural language emotion instructions.
"""

import io
import threading
from pathlib import Path

from openai import OpenAI

from config import TTS_MODEL, TTS_VOICE
from eeg_pipeline import BrainState

try:
    import sounddevice as sd
    import numpy as np
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


class EmotionMapper:
    """Maps EEG brain state to natural language emotion descriptions for TTS."""

    def map_emotion(self, brain_state: BrainState) -> str:
        valence = brain_state.valence
        engagement = brain_state.engagement
        relaxation = brain_state.relaxation
        cognitive_load = brain_state.cognitive_load

        arousal = engagement * 0.6 + (1 - relaxation) * 0.4

        tone_parts = []

        if valence > 0.7:
            if arousal > 0.6:
                tone_parts.append("happy and energetic")
            else:
                tone_parts.append("warm and content")
        elif valence < 0.3:
            if arousal > 0.6:
                tone_parts.append("frustrated or concerned")
            else:
                tone_parts.append("subdued and thoughtful")
        else:
            if arousal > 0.7:
                tone_parts.append("alert and focused")
            elif arousal < 0.3:
                tone_parts.append("calm and relaxed")
            else:
                tone_parts.append("neutral and conversational")

        if engagement > 0.7:
            tone_parts.append("confident")
        elif engagement < 0.3:
            tone_parts.append("gentle and soft-spoken")

        if cognitive_load > 0.6:
            tone_parts.append("deliberate, as if thinking carefully")

        instruction = f"Speak in a {', '.join(tone_parts)} tone. "
        instruction += "Sound natural and human, like talking to a friend. "
        instruction += "Normal conversational pace."

        return instruction


class VoiceOutput:
    """Text-to-speech with emotion-aware voice synthesis."""

    def __init__(self):
        self._client = None
        self._emotion_mapper = EmotionMapper()
        self._speaking = False
        self._last_audio_path = None

    def _get_client(self):
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def speak(self, text: str, brain_state: BrainState = None, blocking=True):
        """Synthesize and play speech with emotion derived from brain state."""
        instructions = "Speak naturally and conversationally."
        if brain_state:
            instructions = self._emotion_mapper.map_emotion(brain_state)

        if blocking:
            self._synthesize_and_play(text, instructions)
        else:
            t = threading.Thread(
                target=self._synthesize_and_play,
                args=(text, instructions),
                daemon=True,
            )
            t.start()

    def _synthesize_and_play(self, text: str, instructions: str):
        self._speaking = True
        try:
            response = self._get_client().audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=text,
                instructions=instructions,
                response_format="pcm",
            )

            audio_bytes = response.read()

            if HAS_AUDIO and len(audio_bytes) > 0:
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
                sd.play(audio_float, samplerate=24000, blocking=True)
            else:
                out_path = Path("neurovoice_output.mp3")
                response_mp3 = self._get_client().audio.speech.create(
                    model=TTS_MODEL,
                    voice=TTS_VOICE,
                    input=text,
                    instructions=instructions,
                )
                response_mp3.stream_to_file(str(out_path))
                self._last_audio_path = out_path
                print(f"[Voice] Saved to {out_path} (install sounddevice for playback)")

        except Exception as e:
            print(f"[Voice] TTS error: {e}")
        finally:
            self._speaking = False

    def speak_simple(self, text: str):
        """Quick speak without emotion mapping."""
        self.speak(text, brain_state=None, blocking=False)

    @property
    def is_speaking(self):
        return self._speaking

    def get_emotion_description(self, brain_state: BrainState) -> str:
        return self._emotion_mapper.map_emotion(brain_state)
