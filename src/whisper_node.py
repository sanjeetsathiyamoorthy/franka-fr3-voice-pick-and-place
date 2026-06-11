#!/usr/bin/env python3
"""
whisper_node.py
===============
VAD-based recording with TTS feedback loop prevention.
Subscribes to /tts_cooldown (Float64) — mutes mic during robot speech.
"""

import threading
import tempfile
import subprocess
import os
import re
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from faster_whisper import WhisperModel
from scipy.signal import resample
import scipy.io.wavfile

ALSA_DEVICE      = 'plughw:0,0'
RECORD_RATE      = 44100
WHISPER_RATE     = 16000
CHUNK_SECONDS    = 1
SILENCE_CHUNKS   = 3
MAX_CHUNKS       = 6
NO_SPEECH_THRESH = 0.65
MIN_TEXT_WORDS   = 2

# ── Dynamic noise-floor calibration ───────────────────────────────────────────
# SPEECH_THRESHOLD is computed at startup from measured background RMS.
# These constants control that process — do NOT hard-code a fixed threshold.
NOISE_CALIBRATION_SECS = 3
NOISE_MULTIPLIER       = 1.3
MIN_SPEECH_THRESHOLD   = 0.015
MAX_SPEECH_THRESHOLD   = 0.5  # CRITICAL: must be > typical noise floor (0.84 in your setup)
MIN_SPEECH_CHUNKS      = 2

HOTWORDS = (
    "pick grab take red green blue zone "
    "one two three four area place put drop "
    "home stop cancel status help fetch retrieve"
)

IGNORED_PHRASES = {
    # Generic filler
    "moving left","moving right","moving up","moving down",
    "returning to default position","done","task complete",
    "thank you","thanks","okay","ok","huh","um","uh",
    # Robot TTS outputs — prevent feedback loop re-recognition
    # Note: "pick red/green/blue" intentionally removed — the LR system needs
    # "pick red and place on the right" to pass through.
    # "picking red/green/blue" (with -ing) still blocks TTS echoes correctly.
    "robot ready","stopping","returning to home position",
    "picking red","picking green","picking blue","picking yellow",
    "placing","retreating","grasping","lifting",
    "say place zone","say zone","zone selected",
    "which block should i pick","which block",
    "cannot see","no block","zone is empty",
    "system idle","system busy","queue empty",
    "left right system ready",
    "picking placing on right side","picking placing on left side",
}

HALLUCINATION_WORDS = {
    'cube','bin','basket','tray','slot','deliver','transfer',
    'reset','park','freeze','halt','visible','detect',
}

CORRECTIONS = {
    r'\bblew\b':        'blue',
    r'\bflew\b':        'blue',
    r'\bclue\b':        'blue',
    r'\bgreed\b':       'green',
    r'\bscreen\b':      'green',
    r'\bclean\b':       'green',
    r'\bread\b':        'red',
    r'\bzone\s+free\b': 'zone three',
    r'\bzone\s+tree\b': 'zone three',
    r'\bzone\s+for\b':  'zone four',
    r'\bzone\s+fore\b': 'zone four',
    r'\bpig\b':         'pick',
    r'\bpit\b':         'pick',
    r'\bpeak\b':        'pick',
    r'\bgrip\b':        'grab',
}

def apply_corrections(text: str) -> str:
    for pattern, fix in CORRECTIONS.items():
        text = re.sub(pattern, fix, text)
    return text

def is_hallucination(text: str) -> bool:
    words = set(text.lower().split())
    return len(words & HALLUCINATION_WORDS) >= 3


class WhisperNode(Node):

    def __init__(self):
        super().__init__('whisper_node')
        self.pub           = self.create_publisher(String, '/voice_raw', 10)
        self._last_text    = ""
        self._ignore_until = 0.0
        self._speech_threshold = MIN_SPEECH_THRESHOLD  # overwritten by calibration
        
        self._robot_home   = True  # Starts off assuming robot is ready/idle

        self.create_subscription(Float64, '/tts_cooldown', self._tts_cb, 10)
        self.create_subscription(String, '/robot_state', self._robot_state_cb, 10)

        self.get_logger().info('Loading Whisper model (small.en)...')
        try:
            self.model = WhisperModel('small.en', device='cpu', compute_type='int8')
            self.get_logger().info('Loaded small.en model')
        except Exception:
            self.get_logger().warn('small.en not found, falling back to small')
            self.model = WhisperModel('small', device='cpu', compute_type='int8')

        # Calibrate noise floor BEFORE starting the listen loop
        self._calibrate_noise_floor()

        self.get_logger().info(
            f'VAD — device={ALSA_DEVICE}  '
            f'threshold={self._speech_threshold:.4f}  '
            f'silence={SILENCE_CHUNKS}s  min_speech_chunks={MIN_SPEECH_CHUNKS}'
        )
        self._running = True
        self._thread  = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        self.get_logger().info('Ready. Listening...')

    # ── Robot state callback ──────────────────────────────────────────────────

    def _robot_state_cb(self, msg: String):
        """Toggle listen capability based on robot mechanical execution state."""
        self._robot_home = (msg.data == "HOME")

    # ── TTS cooldown callback ──────────────────────────────────────────────────

    def _tts_cb(self, msg: Float64):
        """Mute microphone for msg.data seconds to prevent TTS feedback loop."""
        self._ignore_until = time.time() + msg.data
        self.get_logger().info(f'TTS active — mic muted for {msg.data:.1f}s')

    # ── Noise-floor calibration ────────────────────────────────────────────────

    def _calibrate_noise_floor(self):
        """
        Record NOISE_CALIBRATION_SECS seconds of background audio, measure RMS,
        then set self._speech_threshold = noise_floor * NOISE_MULTIPLIER.

        This fixes the core problem: a static threshold of 0.003 is meaningless
        when the real background noise floor is 0.6–0.7 RMS.
        """
        self.get_logger().info(
            f'Calibrating noise floor — keep quiet for {NOISE_CALIBRATION_SECS}s...'
        )
        rms_samples = []
        deadline    = time.time() + NOISE_CALIBRATION_SECS

        while time.time() < deadline:
            chunk = self._record_chunk()
            if chunk is not None:
                rms_samples.append(float(np.sqrt(np.mean(chunk ** 2))))

        if not rms_samples:
            self.get_logger().warn('Calibration failed — using minimum threshold')
            self._speech_threshold = MIN_SPEECH_THRESHOLD
            return

        # Use 95th-percentile so threshold sits above noise bursts, not just average
        noise_peak = float(np.percentile(rms_samples, 95))
        computed   = noise_peak * NOISE_MULTIPLIER
        self._speech_threshold = max(MIN_SPEECH_THRESHOLD, min(computed, MAX_SPEECH_THRESHOLD))

        self.get_logger().info(
            f'Noise peak (p95): {noise_peak:.4f}  →  '
            f'speech threshold: {self._speech_threshold:.4f}'
        )

        if noise_peak > 0.50:
            self.get_logger().warn(
                f'Very high noise floor ({noise_peak:.4f})! '
                f'Check microphone gain or ALSA device. '
                f'Commands may be hard to detect.'
            )

    def _record_chunk(self):
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        cmd = ['arecord', '-D', ALSA_DEVICE, '-d', str(int(CHUNK_SECONDS)),
               '-f', 'S16_LE', '-r', str(RECORD_RATE), '-c', '1', '-q', tmp.name]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            self.get_logger().error(f'arecord: {result.stderr.decode().strip()}')
            os.unlink(tmp.name)
            return None
        try:
            _, data = scipy.io.wavfile.read(tmp.name)
            os.unlink(tmp.name)
            return data.astype(np.float32) / 32768.0
        except Exception as e:
            self.get_logger().error(f'WAV read: {e}')
            os.unlink(tmp.name)
            return None

    def _record_utterance(self):
        chunks              = []
        pre_speech_buffer   = []   # accumulates chunks during streak — flushed on confirmation
        speech_started      = False
        silent_count        = 0
        above_thresh_streak = 0

        while True:
            # Don't start a new utterance while TTS is playing
            if not speech_started and time.time() < self._ignore_until:
                time.sleep(0.2)
                continue

            chunk = self._record_chunk()
            if chunk is None:
                break

            # Abort mid-recording if TTS fires
            if time.time() < self._ignore_until:
                self.get_logger().info('TTS started mid-recording — discarding')
                return None

            rms       = float(np.sqrt(np.mean(chunk ** 2)))
            is_speech = rms >= self._speech_threshold

            if not speech_started:
                if is_speech:
                    above_thresh_streak += 1
                    pre_speech_buffer.append(chunk)   # save — don't discard first chunk
                    if above_thresh_streak >= MIN_SPEECH_CHUNKS:
                        # Confirmed speech — flush all buffered chunks
                        speech_started    = True
                        silent_count      = 0
                        chunks.extend(pre_speech_buffer)
                        pre_speech_buffer = []
                        self.get_logger().info(
                            f'🎙  Recording... (RMS={rms:.4f} '
                            f'threshold={self._speech_threshold:.4f})'
                        )
                else:
                    above_thresh_streak = 0
                    pre_speech_buffer   = []   # reset on silent chunk
            else:
                chunks.append(chunk)
                if is_speech:
                    silent_count = 0
                else:
                    silent_count += 1
                    if silent_count >= SILENCE_CHUNKS:
                        self.get_logger().info('⏹  Processing...')
                        break
                if len(chunks) >= MAX_CHUNKS:
                    self.get_logger().warn('⚠  Max duration — processing...')
                    break

        return np.concatenate(chunks) if chunks else None

    # ── Main listen loop ───────────────────────────────────────────────────────

    def _listen_loop(self):
        self.get_logger().info('Waiting for voice commands...')

        while self._running and rclpy.ok():
            if not self._robot_home:
                time.sleep(0.25)
                continue

            try:
                audio = self._record_utterance()
                if audio is None:
                    continue

                # Final cooldown check (belt-and-suspenders after _record_utterance)
                if time.time() < self._ignore_until:
                    self.get_logger().info('TTS cooldown — discarding audio')
                    continue

                # Reject utterances barely above noise floor
                avg_rms = float(np.sqrt(np.mean(audio ** 2)))
                if avg_rms < self._speech_threshold * 0.8:
                    self.get_logger().debug(f'Audio too quiet (RMS={avg_rms:.4f}) — skipping')
                    continue

                audio = resample(audio, int(len(audio) * WHISPER_RATE / RECORD_RATE))

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    scipy.io.wavfile.write(f.name, WHISPER_RATE, audio)
                    fname = f.name

                segments, _ = self.model.transcribe(
                    fname,
                    language='en',
                    beam_size=7,
                    temperature=0,
                    hotwords=HOTWORDS,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
                    condition_on_previous_text=False,
                )
                os.unlink(fname)

                parts = []
                for seg in segments:
                    if (hasattr(seg, 'no_speech_prob') and
                            seg.no_speech_prob > NO_SPEECH_THRESH):
                        continue
                    if seg.text.strip():
                        parts.append(seg.text.strip())

                text = ' '.join(parts).strip().lower()

                if not text:
                    continue

                words = text.split()

                if len(words) > 15 and len(set(words)) <= 2:
                    self.get_logger().warn(
                        f'Repetitive hallucination discarded: "{text[:80]}"'
                     )
                    continue

                if not text:
                    continue
                if len(text.split()) < MIN_TEXT_WORDS:
                    continue
                if text in IGNORED_PHRASES:
                    continue
                # Reject if text starts with any known robot TTS output
                if any(text.startswith(p) for p in IGNORED_PHRASES if len(p) > 6):
                    self.get_logger().info(f'TTS echo discarded: "{text}"')
                    continue
                if is_hallucination(text):
                    self.get_logger().warn(f'Hallucination discarded: "{text}"')
                    continue

                corrected = apply_corrections(text)
                if corrected != text:
                    self.get_logger().info(f'Corrected: "{text}" → "{corrected}"')
                text = corrected

                if text == self._last_text:
                    continue
                self._last_text = text

                self.get_logger().info(f'✅ Heard: "{text}"')
                msg = String()
                msg.data = text
                self.pub.publish(msg)

            except Exception as e:
                self.get_logger().error(f'Listen loop error: {e}')

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main():
    rclpy.init()
    node = WhisperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()