"""Server-side STT — Continuous mic capture + OpenAI Whisper.

Captures mic audio, detects speech via energy, transcribes with Whisper,
and fires callbacks. Sends live "hearing speech" indicators so the UI
shows activity immediately, not just after transcription completes.
"""

import io
import time
import wave
import threading
import numpy as np
import sounddevice as sd
from openai import OpenAI

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
SILENCE_THRESHOLD = 500
MIN_SPEECH_DURATION = 0.4
SILENCE_GAP = 1.8
PARTIAL_INTERVAL = 2.5


class STTEngine:
    def __init__(self, api_key=None):
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._running = False
        self._thread = None
        self._on_hearing = None
        self._on_partial = None
        self._on_final = None

        self._audio_buffer = []
        self._is_speaking = False
        self._speech_start = 0.0
        self._last_speech = 0.0
        self._last_partial_time = 0.0

    def start(self, on_hearing=None, on_partial=None, on_final=None):
        self._on_hearing = on_hearing
        self._on_partial = on_partial
        self._on_final = on_final
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[STT] Mic capture started", flush=True)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _capture_loop(self):
        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                               blocksize=chunk_samples) as stream:
                while self._running:
                    data, _ = stream.read(chunk_samples)
                    audio_chunk = data[:, 0]
                    energy = float(np.mean(np.abs(audio_chunk)))
                    now = time.time()

                    if energy > SILENCE_THRESHOLD:
                        if not self._is_speaking:
                            self._is_speaking = True
                            self._speech_start = now
                            self._last_partial_time = now
                            self._audio_buffer = []
                            if self._on_hearing:
                                self._on_hearing(True)

                        self._last_speech = now
                        self._audio_buffer.append(audio_chunk.copy())

                        if now - self._last_partial_time >= PARTIAL_INTERVAL and len(self._audio_buffer) > 0:
                            self._last_partial_time = now
                            self._do_partial_indicator()

                    elif self._is_speaking:
                        self._audio_buffer.append(audio_chunk.copy())
                        silence_duration = now - self._last_speech

                        if silence_duration >= SILENCE_GAP:
                            speech_duration = self._last_speech - self._speech_start
                            if speech_duration >= MIN_SPEECH_DURATION and len(self._audio_buffer) > 0:
                                self._do_transcribe(partial=False)
                            self._is_speaking = False
                            self._audio_buffer = []
                            if self._on_hearing:
                                self._on_hearing(False)

        except Exception as e:
            print(f"[STT] Mic error: {e}", flush=True)

    def _do_partial_indicator(self):
        """Show that speech is being captured without transcribing yet."""
        duration = len(self._audio_buffer) * CHUNK_DURATION
        if self._on_partial:
            self._on_partial(f"(listening... {duration:.0f}s)")

    def _do_transcribe(self, partial=False):
        if not self._audio_buffer:
            return

        audio = np.concatenate(self._audio_buffer)

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        buf.seek(0)
        buf.name = 'speech.wav'

        try:
            result = self._client.audio.transcriptions.create(
                model='whisper-1', file=buf, language='en',
            )
            text = result.text.strip()
            if not text:
                return

            if partial:
                if self._on_partial:
                    self._on_partial(text)
            else:
                print(f'[STT] "{text}"', flush=True)
                if self._on_final:
                    self._on_final(text)
        except Exception as e:
            print(f"[STT] Whisper error: {e}", flush=True)

    @property
    def is_listening(self):
        return self._running
