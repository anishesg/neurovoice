"""Voice I/O — AWS Polly TTS with plain text (generative engine)."""

import threading
import numpy as np
import boto3
from eeg_pipeline import BrainState


class VoiceOutput:
    def __init__(self, voice_id="Matthew", region="us-east-1"):
        self._polly = boto3.client("polly", region_name=region)
        self._voice_id = voice_id
        self._speaking = False
        self._sample_rate = 16000

    def speak(self, text, brain=None, blocking=False):
        if blocking:
            self._synth_and_play(text)
        else:
            threading.Thread(target=self._synth_and_play, args=(text,), daemon=True).start()

    def _synth_and_play(self, text):
        self._speaking = True
        try:
            response = self._polly.synthesize_speech(
                Engine="generative",
                VoiceId=self._voice_id,
                OutputFormat="pcm",
                SampleRate=str(self._sample_rate),
                TextType="text",
                Text=text,
            )
            audio_bytes = response["AudioStream"].read()
            if len(audio_bytes) > 0:
                import sounddevice as sd
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                sd.play(audio, samplerate=self._sample_rate, blocking=True)
        except Exception as e:
            print(f"[TTS] Error: {e}", flush=True)
        finally:
            self._speaking = False

    def speak_raw_pcm(self, text, brain=None):
        try:
            response = self._polly.synthesize_speech(
                Engine="generative",
                VoiceId=self._voice_id,
                OutputFormat="pcm",
                SampleRate=str(self._sample_rate),
                TextType="text",
                Text=text,
            )
            return response["AudioStream"].read()
        except Exception as e:
            print(f"[TTS] Error: {e}", flush=True)
            return b""

    def get_emotion_description(self, brain):
        if not brain:
            return "neutral"
        v, e = brain.valence, brain.engagement
        if v > 0.65:
            return "warm, positive" if e < 0.5 else "energetic, happy"
        elif v < 0.35:
            return "subdued" if e < 0.5 else "tense"
        return "calm, neutral"

    @property
    def is_speaking(self):
        return self._speaking
