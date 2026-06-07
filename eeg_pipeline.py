"""EEG Pipeline — Muse S connection, signal processing, brain state.

Consolidated from axiom's connection.py, signal_processing.py, brain_state.py.
Supports real Muse S hardware and simulation mode.
"""

import time
import threading
import numpy as np
from dataclasses import dataclass, field, asdict

from config import (
    BOARD_ID, EEG_SAMPLE_RATE, FREQ_BANDS,
    BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER, EMA_ALPHA,
)

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowPresets
    from brainflow.data_filter import (
        DataFilter, FilterTypes, DetrendOperations,
        NoiseTypes, WindowOperations,
    )
    HAS_BRAINFLOW = True
except ImportError:
    HAS_BRAINFLOW = False

SR = EEG_SAMPLE_RATE
BANDS = list(FREQ_BANDS.items())


@dataclass
class BrainState:
    engagement: float = 0.0
    focus: float = 0.0
    relaxation: float = 0.0
    cognitive_load: float = 0.0
    valence: float = 0.5
    asymmetry: float = 0.0
    jaw_clench: bool = False
    error_response: float = 0.0
    band_powers: dict = field(default_factory=dict)
    band_ratios: dict = field(default_factory=dict)
    signal_quality: list = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self):
        return asdict(self)

    def to_feature_vector(self):
        bp = self.band_powers
        return np.array([
            self.engagement, self.focus, self.valence,
            self.cognitive_load, self.relaxation, self.asymmetry,
            bp.get("alpha", 0) / (bp.get("theta", 1e-6) + 1e-6),
            bp.get("beta", 0) / (bp.get("alpha", 1e-6) + 1e-6),
            bp.get("gamma", 0) / (bp.get("beta", 1e-6) + 1e-6),
            bp.get("theta", 0) / (bp.get("alpha", 1e-6) + 1e-6),
            bp.get("delta", 0) / (bp.get("theta", 1e-6) + 1e-6),
            self.error_response,
        ], dtype=np.float32)


class RingBuffer:
    def __init__(self, channels, max_samples):
        self._buf = np.zeros((channels, max_samples), dtype=np.float64)
        self._cap = max_samples
        self._pos = 0
        self._total = 0
        self._lock = threading.Lock()

    def append(self, data):
        n = data.shape[1]
        if n == 0:
            return
        with self._lock:
            for i in range(n):
                self._buf[:, self._pos % self._cap] = data[:, i]
                self._pos += 1
            self._total += n

    def get_last(self, n):
        with self._lock:
            avail = min(n, self._total, self._cap)
            if avail == 0:
                return np.zeros((self._buf.shape[0], 0))
            end = self._pos % self._cap
            if avail <= end:
                return self._buf[:, end - avail:end].copy()
            return np.concatenate([
                self._buf[:, self._cap - (avail - end):],
                self._buf[:, :end],
            ], axis=1)


class BrainStateEngine:
    def __init__(self):
        self._alpha_bl = None
        self._beta_bl = None
        self._theta_bl = None
        self._clench_cooldown = 0.0
        self._prev_asym = 0.0

    def process(self, eeg_4ch, timestamp=0.0):
        state = BrainState(timestamp=timestamp)
        n = eeg_4ch.shape[1] if eeg_4ch.ndim == 2 else 0
        if n < SR:
            return state

        filtered = np.zeros_like(eeg_4ch, dtype=np.float64)
        for ch in range(4):
            filtered[ch] = self._filter(eeg_4ch[ch])

        ch_powers = [self._band_powers(filtered[ch]) for ch in range(4)]
        avg = {name: float(np.mean([cp[name] for cp in ch_powers])) for name, _ in BANDS}
        state.band_powers = avg

        alpha = avg["alpha"] + 1e-6
        theta = avg["theta"] + 1e-6
        beta = avg["beta"] + 1e-6
        gamma = avg["gamma"] + 1e-6

        if self._alpha_bl is None:
            self._alpha_bl, self._beta_bl, self._theta_bl = alpha, beta, theta
        else:
            e = EMA_ALPHA
            self._alpha_bl = (1 - e) * self._alpha_bl + e * alpha
            self._beta_bl = (1 - e) * self._beta_bl + e * beta
            self._theta_bl = (1 - e) * self._theta_bl + e * theta

        state.band_ratios = {
            "alpha_theta": round(alpha / theta, 4),
            "beta_alpha": round(beta / alpha, 4),
            "gamma_beta": round(gamma / beta, 4),
            "theta_alpha": round(theta / alpha, 4),
        }

        beta_rel = beta / (self._beta_bl + 1e-6)
        state.engagement = float(np.clip((beta_rel - 0.5) / 1.5, 0, 1))
        state.focus = float(np.clip((beta / alpha - 1.0) / 4.0, 0, 1))
        alpha_rel = alpha / (self._alpha_bl + 1e-6)
        state.relaxation = float(np.clip((alpha_rel - 0.5) / 1.5, 0, 1))
        state.cognitive_load = float(np.clip((theta / alpha - 0.5) / 2.0, 0, 1))

        af7_a = ch_powers[1]["alpha"] + 1e-6
        af8_a = ch_powers[2]["alpha"] + 1e-6
        raw_asym = float(np.log(af8_a) - np.log(af7_a))
        state.asymmetry = raw_asym
        state.valence = float(np.clip(0.5 + raw_asym * 0.3, 0, 1))

        state.jaw_clench = self._detect_clench(eeg_4ch, timestamp)

        asym_shift = raw_asym - self._prev_asym
        if asym_shift < -0.3:
            state.error_response = min(1.0, abs(asym_shift))
        self._prev_asym = raw_asym

        state.signal_quality = self._quality(eeg_4ch, filtered)
        return state

    def _filter(self, data):
        if not HAS_BRAINFLOW or len(data) < 12:
            return data.copy()
        out = data.copy()
        DataFilter.detrend(out, DetrendOperations.LINEAR.value)
        DataFilter.perform_bandpass(out, SR, BANDPASS_LOW, BANDPASS_HIGH,
                                    FILTER_ORDER, FilterTypes.BUTTERWORTH.value, 0.0)
        DataFilter.remove_environmental_noise(out, SR, NoiseTypes.SIXTY.value)
        return out

    def _band_powers(self, data):
        if not HAS_BRAINFLOW or len(data) < SR:
            return {name: 0.0 for name, _ in BANDS}
        nfft = DataFilter.get_nearest_power_of_two(SR)
        psd = DataFilter.get_psd_welch(data, nfft, nfft // 2, SR,
                                       WindowOperations.HANNING.value)
        return {name: float(DataFilter.get_band_power(psd, lo, hi))
                for name, (lo, hi) in BANDS}

    def _detect_clench(self, raw, ts):
        if ts < self._clench_cooldown:
            return False
        window = min(25, raw.shape[1])
        tp9 = raw[0, -window:]
        tp10 = raw[3, -window:]
        combined = (np.abs(np.diff(tp9)) + np.abs(np.diff(tp10))) / 2.0
        hf_energy = float(np.mean(combined ** 2))
        is_clench = hf_energy > 200000
        if is_clench:
            self._clench_cooldown = ts + 0.5
        return is_clench

    def _quality(self, raw, filtered):
        quality = []
        for ch in range(4):
            std = float(np.std(raw[ch, -SR:]))
            if std < 1.0 or std > 500.0:
                quality.append(0.0)
            else:
                sig = float(np.mean(filtered[ch, -SR:] ** 2))
                noise = float(np.mean((raw[ch, -SR:] - filtered[ch, -SR:]) ** 2))
                quality.append(min(1.0, (sig / (noise + 1e-10)) / 5.0))
        return quality


class EEGPipeline:
    """Top-level pipeline: manages connection, streaming, and brain state."""

    def __init__(self, simulate=False):
        self.simulate = simulate
        self.ring = RingBuffer(4, SR * 10)
        self.engine = BrainStateEngine()
        self.state = BrainState()
        self._board = None
        self._running = False
        self._sim_phase = 0.0
        self._sim_bias = {"engagement": 0.5, "valence": 0.5}

    def connect(self):
        if self.simulate:
            print("[EEG] Simulation mode — no headband needed")
            return
        if not HAS_BRAINFLOW:
            raise RuntimeError("BrainFlow not installed. Use --simulate.")
        params = BrainFlowInputParams()
        params.timeout = 5
        self._board = BoardShim(BOARD_ID, params)
        print("[EEG] Scanning for Muse S...")
        self._board.prepare_session()
        self._board.start_stream(num_samples=450000)
        self._eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)[:4]
        print("[EEG] Connected, streaming at 256Hz")

    def poll(self):
        if self.simulate:
            self._generate_sim()
        elif self._board:
            raw = self._board.get_board_data(num_samples=64)
            if raw.shape[1] > 0:
                self.ring.append(raw[self._eeg_channels, :])

    def update_brain_state(self):
        window = self.ring.get_last(SR)
        if window.shape[1] >= SR:
            self.state = self.engine.process(window, timestamp=time.time())
        return self.state

    def set_sim_bias(self, engagement=0.5, valence=0.5):
        self._sim_bias = {"engagement": engagement, "valence": valence}

    def _generate_sim(self):
        n = 32
        t = np.arange(n) / SR + self._sim_phase
        self._sim_phase += n / SR
        eng = self._sim_bias["engagement"]
        val = self._sim_bias["valence"]

        eeg = np.zeros((4, n))
        for ch in range(4):
            p = ch * 0.5
            alpha_amp = 15 * (1 - eng * 0.5) + np.random.randn() * 1.5
            eeg[ch] += alpha_amp * np.sin(2 * np.pi * 10 * t + p)
            beta_amp = 8 * (0.3 + eng * 0.7) + np.random.randn() * 1.0
            eeg[ch] += beta_amp * np.sin(2 * np.pi * 22 * t + p * 0.7)
            theta_amp = 10 * (0.5 + (1 - eng) * 0.3) + np.random.randn() * 1.0
            eeg[ch] += theta_amp * np.sin(2 * np.pi * 6 * t + p * 1.2)
            eeg[ch] += np.random.randn(n) * 4

        val_shift = (val - 0.5) * 6
        eeg[1] += val_shift * np.sin(2 * np.pi * 10 * t)
        eeg[2] -= val_shift * np.sin(2 * np.pi * 10 * t)
        self.ring.append(eeg)

    def stop(self):
        if self._board:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception:
                pass
