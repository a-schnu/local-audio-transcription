#!/usr/bin/env python3
""" local transcription + diarization pipeline.

Design goals
------------
- 100% local runtime: offline model resolution by default.
- Stable on 4GB-class NVIDIA GPUs by staging GPU use and defaulting to a
  conservative Whisper profile.
- Modular Sense-Think-Act-Observe workflow:
  1) hardware profiling
  2) signal enhancement + VAD windowing
  3) transcription + diarization
  4) CLEAR-style verification (latency, efficacy, reliability)

This script expects locally available model artifacts. In strict local mode,
Whisper identifiers must already exist in local cache or be explicit local
paths, and the diarization pipeline must be provided as a local path to a
pyannote config.yaml or its containing directory.
"""


from __future__ import annotations

import os
import warnings
import logging

# --- Suppress ALL PyTorch internal logs (including Triton/flop_counter) ---
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["PYTORCH_LOG_LEVEL"] = "ERROR"
os.environ["ABSEIL_LOG_LEVEL"] = "3"

# --- Suppress environment-based warnings ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

# --- Suppress Python warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*triton.*")
warnings.filterwarnings("ignore", message=".*std().*")

# --- Lower log levels for heavy libraries ---
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

logging.getLogger("torch.utils.flop_counter").disabled = True

import argparse
import contextlib
import dataclasses
from dataclasses import dataclass, asdict
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time
import traceback
import uuid
import wave
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import numpy as np
from pyannote.audio import Pipeline




# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HardwareProfile:
    torch_cuda_available: bool
    device: str
    gpu_name: Optional[str]
    vram_gb: Optional[float]
    torch_cuda_version: Optional[str]
    recommended_whisper_model: str
    recommended_compute_type: str
    recommended_diarization_device: str
    recommended_beam_size: int
    recommendation_reason: str


@dataclass
class SpeechWindow:
    """
    Represents a time window in the audio where speech is detected (after VAD).

    Attributes:
        start: Start time of the window in seconds.
        end: End time of the window in seconds.
    """
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class WordStamp:
    """
    Represents a single word with its start/end timestamps and text.
    """
    start: float
    end: float
    text: str


@dataclass
class ASRSegment:
    """
    Represents a segment of transcribed speech from Whisper.

    """
    start: float
    end: float
    text: str
    words: list[WordStamp]


@dataclass
class DiarizationTurn:
    start: float
    end: float
    speaker: str


@dataclass
class Span:
    """
    Represents a final output span: a segment of text attributed to a speaker.

    """
    start: float
    end: float
    text: str
    speaker: str
    original_speaker: str


@dataclass
class PreprocessedAudio:
    """
    Represents audio after preprocessing (enhancement + VAD).
    
    """

    enhanced_path: Path
    waveform: np.ndarray
    sample_rate: int
    duration_sec: float
    windows: list[SpeechWindow]


@dataclass
class RunMetrics:
    """
    Tracks performance metrics for a single pipeline run.

    Attributes:
        preprocessing_sec: Time spent on audio preprocessing (seconds).
        transcription_sec: Time spent on transcription (seconds).
        diarization_sec: Time spent on diarization (seconds).
        alignment_sec: Time spent on aligning transcription and diarization (seconds).
        total_sec: Total runtime (seconds).
        audio_duration_sec: Duration of the input audio (seconds).
        real_time_factor: Ratio of total runtime to audio duration (RTF).
        wer: Word Error Rate (if reference transcript provided).
        diarization_precision: Precision of diarization (if reference RTTM provided).
    """
    preprocessing_sec: float = 0.0
    transcription_sec: float = 0.0
    diarization_sec: float = 0.0
    alignment_sec: float = 0.0
    total_sec: float = 0.0
    audio_duration_sec: float = 0.0
    real_time_factor: float = 0.0
    wer: Optional[float] = None
    diarization_precision: Optional[float] = None


@dataclass
class RunResult:
    """
    Final output of a pipeline run, including transcript, metrics, and metadata.

    """
    transcript: str
    spans: list[Span]
    turns: list[DiarizationTurn]
    metrics: RunMetrics
    detected_language: Optional[str]
    transcript_sha256: str
    whisper_model: str
    diarization_model: str
    transcription_device: str
    diarization_device: str
    compute_type: str


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(message, flush=True)


def error(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
"""
Defines and parses command-line arguments for the pipeline.
This section configures all user-facing options, including:
- Input/output paths (audio, transcripts, reports, state DB).
- Model and device selection (Whisper, diarization, GPU/CPU).
- Preprocessing parameters (VAD, sample rate, chunking).
- Diarization settings (speaker counts, thresholds).
- Verification flags (reference transcripts, consistency runs).
- Safety toggles (online model resolution, VRAM checks).
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Whisper + pyannote diarization pipeline optimized for 4GB-class GPUs."
    )
    parser.add_argument("audio_path", help="Path to local audio/video file.")
    parser.add_argument(
        "--output",
        help="Transcript output path. Default: <input_stem>_transcript.txt",
    )
    parser.add_argument(
        "--report-json",
        help="JSON report path. Default: <input_stem>_report.json",
    )
    parser.add_argument(
        "--state-db",
        help="SQLite state store path. Default: <output_dir>/transcriber_state.sqlite",
    )
    parser.add_argument(
        "--whisper-model", "--model",
        dest="whisper_model",
        default=None,
        help=(
            "Local Faster-Whisper model directory or cached model identifier. "
            "When omitted, the script selects a conservative default from hardware profiling."
        ),
    )
    parser.add_argument(
        "--model-profile",
        default="auto",
        choices=["auto", "latency", "balanced", "accuracy"],
        help="Model-selection policy when --whisper-model is omitted.",
    )
    parser.add_argument(
        "--diarization-config",
        required=True,
        help=(
            "Local path to pyannote diarization config.yaml or a directory containing config.yaml. "
            "This must already exist locally."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Ignored compatibility flag. The optimized script expects a local diarization config instead of pulling models online."
        ),
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code. If omitted, the first speech chunk is used for language detection.",
    )
    parser.add_argument(
        "--transcription-device", "--device",
        dest="transcription_device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for Whisper inference.",
    )
    parser.add_argument(
        "--diarization-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for diarization. On 4GB GPUs, CPU is usually safer.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "float16", "int8_float16", "int8", "float32"],
        help="Whisper compute type. Auto defaults to float16 on CUDA and int8 on CPU.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size for Whisper decoding. Auto selects a hardware-aware default.",
    )
    parser.add_argument(
        "--signal-enhancement",
        default="light",
        choices=["off", "light"],
        help="Audio enhancement profile. 'light' uses mono mixdown, high-pass, RMS normalization, and VAD chunking.",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
        help="Sample rate used for preprocessing and ASR.",
    )
    parser.add_argument(
        "--vad-frame-ms",
        type=int,
        default=30,
        help="Frame size for energy-based VAD.",
    )
    parser.add_argument(
        "--vad-hop-ms",
        type=int,
        default=10,
        help="Hop size for energy-based VAD.",
    )
    parser.add_argument(
        "--vad-padding-ms",
        type=int,
        default=250,
        help="Padding added before and after each detected speech region.",
    )
    parser.add_argument(
        "--vad-threshold-scale",
        type=float,
        default=2.2,
        help="Multiplier applied to estimated noise floor for VAD thresholding.",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=450,
        help="Minimum speech window length to keep.",
    )
    parser.add_argument(
        "--merge-gap-ms",
        type=int,
        default=350,
        help="Merge adjacent VAD windows when the silence gap is below this value.",
    )
    parser.add_argument(
        "--max-chunk-sec",
        type=float,
        default=28.0,
        help="Maximum Whisper chunk duration after VAD merging.",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers for diarization.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Lower bound for diarization speaker count.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Upper bound for diarization speaker count.",
    )
    parser.add_argument(
        "--max-span-gap",
        type=float,
        default=0.8,
        help="Merge adjacent transcription spans from the same speaker when the gap is below this value.",
    )
    parser.add_argument(
        "--consistency-runs",
        type=int,
        default=1,
        help="Number of full pipeline runs for reliability verification. Set to 8 for the requested consistency test.",
    )
    parser.add_argument(
        "--reference-transcript",
        default=None,
        help="Optional reference transcript for WER computation.",
    )
    parser.add_argument(
        "--reference-rttm",
        default=None,
        help="Optional reference RTTM for approximate diarization precision evaluation.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep preprocessed audio and chunk WAV files for inspection.",
    )
    parser.add_argument(
        "--allow-online-model-resolution",
        action="store_true",
        help="Allow online model resolution. Disabled by default to keep runtime fully local.",
    )
    parser.add_argument(
        "--force-risky-model",
        action="store_true",
        help="Bypass 4GB VRAM safety checks for medium/turbo/large GPU inference.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment and hardware
# ---------------------------------------------------------------------------
"""

Utilities for configuring the runtime environment and profiling hardware.

"""

#- `configure_offline_mode()`: Enforces offline mode for all dependencies (Hugging Face, Transformers, etc.).

def configure_offline_mode(allow_online: bool) -> None:
    if allow_online:
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")


#`resolve_output_paths()`: Resolves output paths for transcripts, reports, and state DB.

def resolve_output_paths(audio_path: Path, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = audio_path.with_name(f"{audio_path.stem}_transcript.txt")
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
    else:
        report_path = audio_path.with_name(f"{audio_path.stem}_report.json")
    if args.state_db:
        state_db = Path(args.state_db).expanduser().resolve()
    else:
        state_db = output_path.with_name("transcriber_state.sqlite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    state_db.parent.mkdir(parents=True, exist_ok=True)
    return output_path, report_path, state_db

#`probe_hardware()`: Detects GPU/CPU capabilities and recommends optimal settings (model size, compute type, beam size).
def probe_hardware(
    transcription_device: str,
    diarization_device: str,
    model_profile: str,
    beam_size_override: Optional[int],
) -> HardwareProfile:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for this pipeline.") from exc

    cuda_available = torch.cuda.is_available()
    gpu_name: Optional[str] = None
    vram_gb: Optional[float] = None
    torch_cuda_version = getattr(torch.version, "cuda", None)

    if transcription_device == "auto":
        resolved_device = "cuda" if cuda_available else "cpu"
    else:
        resolved_device = transcription_device

    if resolved_device == "cuda" and not cuda_available:
        raise RuntimeError("--transcription-device cuda was requested, but CUDA is unavailable.")

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_gb = round(props.total_memory / (1024 ** 3), 2)

    recommended_model = "small"
    recommended_compute_type = "float16" if resolved_device == "cuda" else "int8"
    recommended_diarization_device = "cpu"
    recommended_beam_size = 2
    reason = "4GB-class GPU profile defaults to Whisper small fp16 to avoid OOM while keeping acceptable accuracy."

    if resolved_device == "cpu":
        recommended_model = "small"
        recommended_compute_type = "int8"
        recommended_diarization_device = "cpu"
        recommended_beam_size = 3
        reason = "CPU-only fallback prioritizes small int8 for bounded RAM and predictable latency."
    elif vram_gb is not None:
        if vram_gb <= 4.5:
            recommended_model = "small"
            recommended_compute_type = "float16"
            recommended_diarization_device = "cpu"
            recommended_beam_size = 2
            reason = (
                "4GB VRAM leaves little headroom for medium/turbo in pure fp16 once audio features, decoder state, "
                "and diarization overhead are considered."
            )
        elif vram_gb <= 6.5:
            recommended_model = "turbo" if model_profile == "latency" else "small"
            recommended_compute_type = "float16"
            recommended_diarization_device = "cpu" if model_profile != "latency" else "cuda"
            recommended_beam_size = 1 if model_profile == "latency" else 3
            reason = "Mid-range VRAM can sustain turbo or small fp16, but CPU diarization still avoids GPU contention."
        else:
            if model_profile == "accuracy":
                recommended_model = "medium"
                recommended_beam_size = 4
            elif model_profile == "latency":
                recommended_model = "turbo"
                recommended_beam_size = 1
            else:
                recommended_model = "turbo"
                recommended_beam_size = 2
            recommended_compute_type = "float16"
            recommended_diarization_device = "cuda"
            reason = "Higher VRAM allows faster or larger fp16 Whisper profiles and GPU diarization without aggressive staging."

    if beam_size_override is not None:
        recommended_beam_size = max(1, int(beam_size_override))

    if diarization_device != "auto":
        recommended_diarization_device = diarization_device

    return HardwareProfile(
        torch_cuda_available=cuda_available,
        device=resolved_device,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        torch_cuda_version=torch_cuda_version,
        recommended_whisper_model=recommended_model,
        recommended_compute_type=recommended_compute_type,
        recommended_diarization_device=recommended_diarization_device,
        recommended_beam_size=recommended_beam_size,
        recommendation_reason=reason,
    )



'''
`choose_*()`: Helper functions to select Whisper model, compute type, diarization device, and beam size based on args/hardware.
'''

def choose_whisper_model(args: argparse.Namespace, hardware: HardwareProfile) -> str:
    if args.whisper_model:
        return args.whisper_model
    if args.model_profile == "latency" and hardware.device == "cuda" and (hardware.vram_gb or 0.0) > 5.5:
        return "turbo"
    if args.model_profile == "accuracy" and hardware.device == "cuda" and (hardware.vram_gb or 0.0) > 6.5:
        return "medium"
    return hardware.recommended_whisper_model


def choose_compute_type(args: argparse.Namespace, hardware: HardwareProfile) -> str:
    if args.compute_type != "auto":
        return args.compute_type
    return hardware.recommended_compute_type


def choose_diarization_device(args: argparse.Namespace, hardware: HardwareProfile) -> str:
    if args.diarization_device != "auto":
        return args.diarization_device
    return hardware.recommended_diarization_device


def choose_beam_size(args: argparse.Namespace, hardware: HardwareProfile) -> int:
    if args.beam_size is not None:
        return max(1, int(args.beam_size))
    return hardware.recommended_beam_size

#`maybe_warn_on_cuda13_stack()`: Warns if CUDA 13.x is detected (potential compatibility issues).

def maybe_warn_on_cuda13_stack(hardware: HardwareProfile) -> None:
    if hardware.device != "cuda":
        return
    if hardware.torch_cuda_version and hardware.torch_cuda_version.startswith("13"):
        log(
            "Warning: the Python GPU stack reports CUDA 13.x. Verify faster-whisper/CTranslate2 compatibility in your local environment."
        )

#`validate_model_choice()`: Validates that the selected Whisper model is safe for the detected hardware (VRAM checks).

def validate_model_choice(
    whisper_model: str,
    compute_type: str,
    hardware: HardwareProfile,
    force_risky_model: bool,
) -> None:
    if hardware.device != "cuda":
        return
    risky_markers = ("medium", "turbo", "large")
    lowered = whisper_model.lower()
    if hardware.vram_gb is not None and hardware.vram_gb <= 4.5 and compute_type == "float16":
        if any(marker in lowered for marker in risky_markers) and not force_risky_model:
            raise RuntimeError(
                "Selected Whisper model is risky for a 4GB GPU in pure fp16. "
                "Use --whisper-model small, switch to --compute-type int8_float16, or pass --force-risky-model if you accept OOM risk."
            )


# ---------------------------------------------------------------------------
# SQLite state store
# ---------------------------------------------------------------------------
"""
SQLite database utilities for tracking pipeline runs, speaker aliases, and utterances.
- `init_state_db()`: Initializes the SQLite database with tables for runs, speaker aliases, and utterances.
- `sanitize_for_json()`: Converts complex Python objects (e.g., Path, dataclass) to JSON-serializable formats.
- `load_alias_map()` / `persist_alias_map()`: Loads/saves speaker alias mappings for consistency across runs.
- `record_run_*()`: Logs run start/end, success/failure, and metrics to the database.
- `persist_utterances()`: Stores final utterances (spans) for a run, linked to the run ID.
"""


def init_state_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            run_group_id TEXT,
            run_index INTEGER,
            started_at TEXT,
            completed_at TEXT,
            audio_path TEXT,
            audio_sha256 TEXT,
            whisper_model TEXT,
            diarization_model TEXT,
            transcription_device TEXT,
            diarization_device TEXT,
            compute_type TEXT,
            audio_duration_sec REAL,
            preprocessing_sec REAL,
            transcription_sec REAL,
            diarization_sec REAL,
            alignment_sec REAL,
            total_sec REAL,
            real_time_factor REAL,
            wer REAL,
            diarization_precision REAL,
            transcript_sha256 TEXT,
            status TEXT,
            error_text TEXT,
            output_path TEXT,
            report_path TEXT,
            config_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS speaker_aliases (
            audio_sha256 TEXT,
            original_speaker TEXT,
            display_speaker TEXT,
            PRIMARY KEY (audio_sha256, original_speaker)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS utterances (
            utterance_id TEXT PRIMARY KEY,
            run_id TEXT,
            idx INTEGER,
            original_speaker TEXT,
            display_speaker TEXT,
            start_sec REAL,
            end_sec REAL,
            text TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        )
        """
    )
    conn.commit()
    return conn


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return sanitize_for_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    return value


def load_alias_map(conn: sqlite3.Connection, audio_sha256: str) -> OrderedDict[str, str]:
    rows = conn.execute(
        "SELECT original_speaker, display_speaker FROM speaker_aliases WHERE audio_sha256 = ? ORDER BY display_speaker",
        (audio_sha256,),
    ).fetchall()
    mapping: OrderedDict[str, str] = OrderedDict()
    for original_speaker, display_speaker in rows:
        mapping[str(original_speaker)] = str(display_speaker)
    return mapping


def persist_alias_map(conn: sqlite3.Connection, audio_sha256: str, alias_map: OrderedDict[str, str]) -> None:
    for original_speaker, display_speaker in alias_map.items():
        conn.execute(
            """
            INSERT INTO speaker_aliases (audio_sha256, original_speaker, display_speaker)
            VALUES (?, ?, ?)
            ON CONFLICT(audio_sha256, original_speaker)
            DO UPDATE SET display_speaker = excluded.display_speaker
            """,
            (audio_sha256, original_speaker, display_speaker),
        )
    conn.commit()


def record_run_start(
    conn: sqlite3.Connection,
    run_id: str,
    run_group_id: str,
    run_index: int,
    audio_path: Path,
    audio_sha256: str,
    whisper_model: str,
    diarization_model: str,
    transcription_device: str,
    diarization_device: str,
    compute_type: str,
    output_path: Path,
    report_path: Path,
    args: argparse.Namespace,
) -> None:
    conn.execute(
        """
        INSERT INTO runs (
            run_id, run_group_id, run_index, started_at, audio_path, audio_sha256,
            whisper_model, diarization_model, transcription_device, diarization_device,
            compute_type, status, output_path, report_path, config_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            run_group_id,
            run_index,
            utc_now_iso(),
            str(audio_path),
            audio_sha256,
            whisper_model,
            diarization_model,
            transcription_device,
            diarization_device,
            compute_type,
            "running",
            str(output_path),
            str(report_path),
            json.dumps(sanitize_for_json(vars(args)), ensure_ascii=False, indent=2),
        ),
    )
    conn.commit()


def record_run_success(
    conn: sqlite3.Connection,
    run_id: str,
    result: RunResult,
) -> None:
    metrics = result.metrics
    conn.execute(
        """
        UPDATE runs SET
            completed_at = ?,
            audio_duration_sec = ?,
            preprocessing_sec = ?,
            transcription_sec = ?,
            diarization_sec = ?,
            alignment_sec = ?,
            total_sec = ?,
            real_time_factor = ?,
            wer = ?,
            diarization_precision = ?,
            transcript_sha256 = ?,
            status = ?
        WHERE run_id = ?
        """,
        (
            utc_now_iso(),
            metrics.audio_duration_sec,
            metrics.preprocessing_sec,
            metrics.transcription_sec,
            metrics.diarization_sec,
            metrics.alignment_sec,
            metrics.total_sec,
            metrics.real_time_factor,
            metrics.wer,
            metrics.diarization_precision,
            result.transcript_sha256,
            "success",
            run_id,
        ),
    )
    conn.commit()


def record_run_failure(conn: sqlite3.Connection, run_id: str, exc: BaseException) -> None:
    conn.execute(
        "UPDATE runs SET completed_at = ?, status = ?, error_text = ? WHERE run_id = ?",
        (utc_now_iso(), "failed", f"{type(exc).__name__}: {exc}", run_id),
    )
    conn.commit()


def persist_utterances(
    conn: sqlite3.Connection,
    run_id: str,
    spans: list[Span],
) -> None:
    conn.execute("DELETE FROM utterances WHERE run_id = ?", (run_id,))
    for idx, span in enumerate(spans):
        conn.execute(
            """
            INSERT INTO utterances (
                utterance_id, run_id, idx, original_speaker, display_speaker,
                start_sec, end_sec, text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                run_id,
                idx,
                span.original_speaker,
                span.speaker,
                span.start,
                span.end,
                span.text,
            ),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# File helpers and transcript formatting
# ---------------------------------------------------------------------------


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_ts(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).strip().split())


def render_transcript(spans: list[Span]) -> str:
    blocks: list[str] = []
    for span in spans:
        blocks.append(f"[{span.speaker}]")
        blocks.append(f"[{format_ts(span.start)} - {format_ts(span.end)}]")
        blocks.append(span.text)
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def write_rttm(turns: list[DiarizationTurn], output_path: Path, file_id: str) -> None:
    lines = []
    for turn in turns:
        duration = max(0.0, turn.end - turn.start)
        lines.append(
            f"SPEAKER {file_id} 1 {turn.start:.3f} {duration:.3f} <NA> <NA> {turn.speaker} <NA> <NA>"
        )
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Audio decoding, enhancement, VAD, chunking
# ---------------------------------------------------------------------------

"""
Utilities for preprocessing audio before transcription and diarization.

"""
#`decode_audio_to_mono()`: Decodes audio/video files into a mono waveform at the target sample rate.
def decode_audio_to_mono(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    try:
        import av
    except Exception as exc:
        raise RuntimeError("PyAV is required. Install 'av' to decode local audio/video files.") from exc

    with av.open(str(path)) as container:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise RuntimeError(f"No audio stream found in file: {path}")
        stream = audio_streams[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            resampled = resampler.resample(frame)
            if resampled is None:
                continue
            frames = resampled if isinstance(resampled, list) else [resampled]
            for out_frame in frames:
                arr = out_frame.to_ndarray()
                if arr.ndim == 2:
                    arr = arr.reshape(-1)
                chunks.append(arr.astype(np.int16, copy=False))

        flushed = resampler.resample(None)
        if flushed is not None:
            frames = flushed if isinstance(flushed, list) else [flushed]
            for out_frame in frames:
                arr = out_frame.to_ndarray()
                if arr.ndim == 2:
                    arr = arr.reshape(-1)
                chunks.append(arr.astype(np.int16, copy=False))

    if not chunks:
        raise RuntimeError("Decoded audio is empty.")

    waveform = np.concatenate(chunks).astype(np.float32) / 32768.0
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    return waveform, sample_rate

#`high_pass_filter()`: Applies a high-pass filter to remove low-frequency noise.
def high_pass_filter(samples: np.ndarray, sample_rate: int, cutoff_hz: float = 80.0) -> np.ndarray:
    if samples.size < 2:
        return samples.copy()
    dt = 1.0 / float(sample_rate)
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = rc / (rc + dt)
    out = np.empty_like(samples)
    out[0] = samples[0]
    prev_y = out[0]
    prev_x = samples[0]
    for i in range(1, samples.size):
        x = float(samples[i])
        y = alpha * (prev_y + x - prev_x)
        out[i] = y
        prev_y = y
        prev_x = x
    return out

#`normalize_rms()`: Normalizes audio to a target RMS level (dBFS) while avoiding clipping.
def normalize_rms(samples: np.ndarray, target_dbfs: float = -20.0, peak_limit: float = 0.98) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32)) + 1e-10))
    if rms <= 1e-6:
        return samples.copy()
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = target_rms / rms
    scaled = samples * gain
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak > peak_limit:
        scaled = scaled * (peak_limit / peak)
    return np.clip(scaled, -1.0, 1.0).astype(np.float32, copy=False)

#`enhance_signal()`: Applies light signal enhancement (high-pass + RMS normalization).
def enhance_signal(samples: np.ndarray, sample_rate: int, mode: str) -> np.ndarray:
    if mode == "off":
        return samples.astype(np.float32, copy=True)
    centered = samples - float(np.mean(samples))
    filtered = high_pass_filter(centered.astype(np.float32, copy=False), sample_rate)
    normalized = normalize_rms(filtered)
    return normalized.astype(np.float32, copy=False)

#`frame_rms()`: Computes RMS energy for sliding windows (used for VAD).
def frame_rms(samples: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    if samples.size < frame_len:
        padded = np.pad(samples, (0, frame_len - samples.size))
        return np.array([math.sqrt(float(np.mean(np.square(padded))) + 1e-10)], dtype=np.float32)
    frame_count = 1 + int(math.ceil((samples.size - frame_len) / float(hop_len)))
    padded_len = (frame_count - 1) * hop_len + frame_len
    padded = np.pad(samples, (0, max(0, padded_len - samples.size)))
    stride = padded.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(frame_count, frame_len),
        strides=(hop_len * stride, stride),
        writeable=False,
    )
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-10, dtype=np.float32)

#`detect_speech_windows()`: Performs Voice Activity Detection (VAD) to split audio into speech/non-speech segments.
def detect_speech_windows(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int,
    hop_ms: int,
    padding_ms: int,
    threshold_scale: float,
    min_speech_ms: int,
    merge_gap_ms: int,
    max_chunk_sec: float,
) -> list[SpeechWindow]:
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, int(sample_rate * hop_ms / 1000.0))
    energy = frame_rms(samples, frame_len, hop_len)
    if energy.size == 0:
        return [SpeechWindow(0.0, 0.0)]

    nonzero = energy[energy > 1e-7]
    if nonzero.size == 0:
        return [SpeechWindow(0.0, len(samples) / float(sample_rate))]

    noise_floor = float(np.percentile(nonzero, 20))
    threshold = max(noise_floor * threshold_scale, 0.0035)
    active = energy >= threshold

    pad_frames = max(0, int(round(padding_ms / float(hop_ms))))
    if pad_frames > 0:
        kernel = np.ones(2 * pad_frames + 1, dtype=np.int32)
        active = np.convolve(active.astype(np.int32), kernel, mode="same") > 0

    min_frames = max(1, int(round(min_speech_ms / float(hop_ms))))
    merge_gap_frames = max(0, int(round(merge_gap_ms / float(hop_ms))))

    windows: list[SpeechWindow] = []
    start_idx: Optional[int] = None
    for i, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = i
        elif not is_active and start_idx is not None:
            end_idx = i
            if end_idx - start_idx >= min_frames:
                windows.append(
                    SpeechWindow(
                        start=start_idx * hop_len / float(sample_rate),
                        end=min(len(samples), end_idx * hop_len + frame_len) / float(sample_rate),
                    )
                )
            start_idx = None
    if start_idx is not None:
        end_idx = len(active)
        if end_idx - start_idx >= min_frames:
            windows.append(
                SpeechWindow(
                    start=start_idx * hop_len / float(sample_rate),
                    end=min(len(samples), end_idx * hop_len + frame_len) / float(sample_rate),
                )
            )

    if not windows:
        return [SpeechWindow(0.0, len(samples) / float(sample_rate))]

    merged: list[SpeechWindow] = [windows[0]]
    for win in windows[1:]:
        last = merged[-1]
        gap_frames = int(round((win.start - last.end) * sample_rate / float(hop_len)))
        combined_duration = win.end - last.start
        if gap_frames <= merge_gap_frames and combined_duration <= max_chunk_sec:
            merged[-1] = SpeechWindow(start=last.start, end=max(last.end, win.end))
        else:
            merged.append(win)

    # Split overlong windows to keep Whisper latency stable.
    final_windows: list[SpeechWindow] = []
    for win in merged:
        if win.duration <= max_chunk_sec:
            final_windows.append(win)
            continue
        cursor = win.start
        while cursor < win.end:
            chunk_end = min(win.end, cursor + max_chunk_sec)
            final_windows.append(SpeechWindow(start=cursor, end=chunk_end))
            cursor = chunk_end
    return final_windows

#`write_pcm16_wav()`: Writes a waveform to a 16-bit PCM WAV file.
def write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

#`preprocess_audio()`: Orchestrates the full preprocessing pipeline (decoding, enhancement, VAD, chunking).
def preprocess_audio(
    audio_path: Path,
    temp_dir: Path,
    args: argparse.Namespace,
) -> PreprocessedAudio:
    waveform, sample_rate = decode_audio_to_mono(audio_path, args.target_sample_rate)
    enhanced = enhance_signal(waveform, sample_rate, args.signal_enhancement)
    windows = detect_speech_windows(
        samples=enhanced,
        sample_rate=sample_rate,
        frame_ms=args.vad_frame_ms,
        hop_ms=args.vad_hop_ms,
        padding_ms=args.vad_padding_ms,
        threshold_scale=args.vad_threshold_scale,
        min_speech_ms=args.min_speech_ms,
        merge_gap_ms=args.merge_gap_ms,
        max_chunk_sec=args.max_chunk_sec,
    )
    enhanced_path = temp_dir / "enhanced_full.wav"
    write_pcm16_wav(enhanced_path, enhanced, sample_rate)
    return PreprocessedAudio(
        enhanced_path=enhanced_path,
        waveform=enhanced,
        sample_rate=sample_rate,
        duration_sec=(enhanced.size / float(sample_rate)) if sample_rate > 0 else 0.0,
        windows=windows,
    )


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------


def cleanup_cuda_cache() -> None:
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def build_whisper_model(
    whisper_model: str,
    device: str,
    compute_type: str,
    strict_local: bool,
):
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError("faster-whisper is required for transcription.") from exc

    kwargs: dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
    }

    init_sig = inspect.signature(WhisperModel.__init__)
    if "cpu_threads" in init_sig.parameters:
        kwargs["cpu_threads"] = max(1, min(8, (os.cpu_count() or 4)))
    if "num_workers" in init_sig.parameters:
        kwargs["num_workers"] = 1
    if strict_local and "local_files_only" in init_sig.parameters:
        kwargs["local_files_only"] = True

    return WhisperModel(whisper_model, **kwargs)


def transcribe_chunk_file(
    model: Any,
    chunk_path: Path,
    start_offset: float,
    language: Optional[str],
    beam_size: int,
) -> tuple[list[ASRSegment], Optional[str]]:
    transcribe_sig = inspect.signature(model.transcribe)
    kwargs: dict[str, Any] = {
        "language": language,
        "beam_size": beam_size,
        "word_timestamps": True,
        "condition_on_previous_text": False,
    }
    if "vad_filter" in transcribe_sig.parameters:
        kwargs["vad_filter"] = False

    segments_iter, info = model.transcribe(str(chunk_path), **kwargs)
    segments = list(segments_iter)
    detected_language = getattr(info, "language", None)

    out_segments: list[ASRSegment] = []
    for seg in segments:
        seg_text = normalize_whitespace(getattr(seg, "text", ""))
        if not seg_text:
            continue
        seg_start = float(getattr(seg, "start", 0.0)) + start_offset
        seg_end = float(getattr(seg, "end", seg_start)) + start_offset
        words: list[WordStamp] = []
        for word in getattr(seg, "words", None) or []:
            word_text = normalize_whitespace(getattr(word, "word", ""))
            word_start = getattr(word, "start", None)
            word_end = getattr(word, "end", None)
            if not word_text or word_start is None or word_end is None:
                continue
            words.append(
                WordStamp(
                    start=float(word_start) + start_offset,
                    end=float(word_end) + start_offset,
                    text=word_text,
                )
            )
        out_segments.append(ASRSegment(start=seg_start, end=seg_end, text=seg_text, words=words))

    return out_segments, detected_language


def transcribe_windows(
    preprocessed: PreprocessedAudio,
    whisper_model: str,
    device: str,
    compute_type: str,
    language: Optional[str],
    beam_size: int,
    keep_intermediates: bool,
    strict_local: bool,
) -> tuple[list[ASRSegment], Optional[str]]:

    model = build_whisper_model(
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        strict_local=strict_local,
    )

    detected_language = language
    all_segments: list[ASRSegment] = []
    chunk_root_obj: Optional[tempfile.TemporaryDirectory] = None
    if keep_intermediates:
        chunk_root = preprocessed.enhanced_path.parent / f"chunks_{int(time.time() * 1000)}"
        chunk_root.mkdir(parents=True, exist_ok=True)
    else:
        chunk_root_obj = tempfile.TemporaryDirectory(prefix="fw_chunks_")
        chunk_root = Path(chunk_root_obj.name)
    try:
        for idx, window in enumerate(preprocessed.windows):
            start_idx = max(0, int(round(window.start * preprocessed.sample_rate)))
            end_idx = min(preprocessed.waveform.size, int(round(window.end * preprocessed.sample_rate)))
            if end_idx <= start_idx:
                continue
            chunk = preprocessed.waveform[start_idx:end_idx]
            chunk_path = chunk_root / f"chunk_{idx:05d}.wav"
            write_pcm16_wav(chunk_path, chunk, preprocessed.sample_rate)
            chunk_segments, chunk_language = transcribe_chunk_file(
                model=model,
                chunk_path=chunk_path,
                start_offset=window.start,
                language=detected_language,
                beam_size=beam_size,
            )
            if detected_language is None and chunk_language:
                detected_language = str(chunk_language)
            all_segments.extend(chunk_segments)
            if not keep_intermediates:
                with contextlib.suppress(Exception):
                    chunk_path.unlink()
    finally:
        del model
        cleanup_cuda_cache()
        if chunk_root_obj is not None:
            chunk_root_obj.cleanup()

    all_segments.sort(key=lambda s: (s.start, s.end))
    return all_segments, detected_language


# ---------------------------------------------------------------------------
# Diarization
# ---------------------------------------------------------------------------

"""
Utilities for speaker diarization using pyannote.
- `resolve_diarization_origin()`: Resolves the diarization model path/config.
- `diarize_audio()`: Runs pyannote diarization on preprocessed audio and returns speaker turns.
"""

def resolve_diarization_origin(config_arg: str) -> str:
    return config_arg
    

def diarize_audio(
    preprocessed: PreprocessedAudio,
    diarization_origin: str,
    device: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> list[DiarizationTurn]:
    warnings.filterwarnings(
        "ignore",
        message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
        category=UserWarning,
    )
    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise RuntimeError("pyannote.audio is required for diarization.") from exc

    pipeline = pipeline = Pipeline.from_pretrained(diarization_origin, token=os.environ.get("HF_TOKEN"))
    if device == "cuda":
        import torch

        pipeline.to(torch.device("cuda"))

    diarization_kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        diarization_kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers is not None:
            diarization_kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            diarization_kwargs["max_speakers"] = int(max_speakers)

    import torch

    diarization_input = {
        "waveform": torch.from_numpy(np.ascontiguousarray(preprocessed.waveform)).unsqueeze(0).float(),
        "sample_rate": int(preprocessed.sample_rate),
    }
    output = pipeline(diarization_input, **diarization_kwargs)
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    turns: list[DiarizationTurn] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            DiarizationTurn(
                start=float(turn.start),
                end=float(turn.end),
                speaker=str(speaker),
            )
        )
    turns.sort(key=lambda t: (t.start, t.end, t.speaker))

    del pipeline
    cleanup_cuda_cache()
    return turns


# ---------------------------------------------------------------------------
# Alignment and speaker labeling
# ---------------------------------------------------------------------------

"""
Utilities for aligning transcription segments with diarization turns and labeling speakers.
- `overlap_seconds()`: Computes the overlap duration between two time intervals.
- `nearest_speaker()`: Assigns a speaker to a segment based on the nearest diarization turn.
- `assign_speaker()`: Assigns a speaker to a segment based on the highest overlap with diarization turns.
- `build_word_level_spans()`: Aligns word-level transcription segments with diarization turns.
- `build_segment_level_spans()`: Aligns segment-level transcription with diarization turns.
- `relabel_speakers()`: Maps original diarization speaker labels to user-friendly labels (e.g., "SPEAKER_00" → "Speaker 1").
- `merge_adjacent_spans()`: Merges adjacent spans from the same speaker to reduce fragmentation.
"""


def overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def nearest_speaker(start: float, end: float, turns: list[DiarizationTurn]) -> str:
    midpoint = (start + end) / 2.0
    best_speaker = "UNKNOWN"
    best_distance = float("inf")
    for turn in turns:
        if turn.start <= midpoint <= turn.end:
            return turn.speaker
        distance = min(abs(midpoint - turn.start), abs(midpoint - turn.end))
        if distance < best_distance:
            best_distance = distance
            best_speaker = turn.speaker
    return best_speaker


def assign_speaker(start: float, end: float, turns: list[DiarizationTurn]) -> str:
    best_speaker = None
    best_overlap = 0.0
    for turn in turns:
        ov = overlap_seconds(start, end, turn.start, turn.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = turn.speaker
    if best_speaker is not None:
        return best_speaker
    return nearest_speaker(start, end, turns)


def build_word_level_spans(segments: list[ASRSegment], turns: list[DiarizationTurn], max_gap: float) -> list[Span]:
    words: list[tuple[float, float, str, str]] = []
    for segment in segments:
        for word in segment.words:
            if not word.text:
                continue
            speaker = assign_speaker(word.start, word.end, turns)
            words.append((word.start, word.end, word.text, speaker))
    if not words:
        return []

    spans: list[Span] = []
    cur_start, cur_end, cur_text, cur_speaker = words[0]
    for word_start, word_end, word_text, speaker in words[1:]:
        same_speaker = speaker == cur_speaker
        gap_ok = (word_start - cur_end) <= max_gap
        if same_speaker and gap_ok:
            cur_end = word_end
            cur_text = normalize_whitespace(f"{cur_text} {word_text}")
        else:
            spans.append(
                Span(
                    start=cur_start,
                    end=cur_end,
                    text=normalize_whitespace(cur_text),
                    speaker=cur_speaker,
                    original_speaker=cur_speaker,
                )
            )
            cur_start, cur_end, cur_text, cur_speaker = word_start, word_end, word_text, speaker
    spans.append(
        Span(
            start=cur_start,
            end=cur_end,
            text=normalize_whitespace(cur_text),
            speaker=cur_speaker,
            original_speaker=cur_speaker,
        )
    )
    return spans


def build_segment_level_spans(segments: list[ASRSegment], turns: list[DiarizationTurn]) -> list[Span]:
    spans: list[Span] = []
    for segment in segments:
        text = normalize_whitespace(segment.text)
        if not text:
            continue
        speaker = assign_speaker(segment.start, segment.end, turns)
        spans.append(
            Span(
                start=segment.start,
                end=segment.end,
                text=text,
                speaker=speaker,
                original_speaker=speaker,
            )
        )
    return spans


def relabel_speakers(
    spans: list[Span],
    persisted_aliases: OrderedDict[str, str],
) -> tuple[list[Span], OrderedDict[str, str]]:
    alias_map: OrderedDict[str, str] = OrderedDict(persisted_aliases)
    next_index = 1
    existing_labels = set(alias_map.values())
    while f"Speaker {next_index}" in existing_labels:
        next_index += 1

    relabeled: list[Span] = []
    for span in spans:
        if span.original_speaker not in alias_map:
            alias_map[span.original_speaker] = f"Speaker {next_index}"
            next_index += 1
        relabeled.append(
            Span(
                start=span.start,
                end=span.end,
                text=span.text,
                speaker=alias_map[span.original_speaker],
                original_speaker=span.original_speaker,
            )
        )
    return relabeled, alias_map


def merge_adjacent_spans(spans: list[Span], max_gap: float) -> list[Span]:
    if not spans:
        return []
    merged: list[Span] = [spans[0]]
    for span in spans[1:]:
        last = merged[-1]
        if span.speaker == last.speaker and (span.start - last.end) <= max_gap:
            merged[-1] = Span(
                start=last.start,
                end=max(last.end, span.end),
                text=normalize_whitespace(f"{last.text} {span.text}"),
                speaker=last.speaker,
                original_speaker=last.original_speaker,
            )
        else:
            merged.append(span)
    return merged


# ---------------------------------------------------------------------------
# Metrics and evaluation
# ---------------------------------------------------------------------------

"""
Utilities for computing evaluation metrics (WER, diarization precision).
- `normalize_for_wer()`: Normalizes text for Word Error Rate (WER) computation.
- `extract_reference_transcript_text()`: Extracts reference text from a transcript file.
- `compute_wer()`: Computes WER between reference and hypothesis transcripts.
- `parse_rttm()`: Parses an RTTM file into a list of diarization turns.
- `speaker_overlap_matrix()`: Computes overlap between predicted and reference speaker turns.
- `best_speaker_mapping()`: Finds the optimal mapping between predicted and reference speakers.
- `compute_diarization_precision()`: Computes diarization precision using speaker overlap.
"""

def normalize_for_wer(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^\w\s']+", " ", text)
    return [tok for tok in text.split() if tok]


def extract_reference_transcript_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\[[^\]]+\]", stripped):
            continue
        lines.append(stripped)
    return " ".join(lines)


def compute_wer(reference_text: str, hypothesis_text: str) -> Optional[float]:
    ref = normalize_for_wer(reference_text)
    hyp = normalize_for_wer(hypothesis_text)
    if not ref:
        return None

    prev = list(range(len(hyp) + 1))
    for i, ref_tok in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, hyp_tok in enumerate(hyp, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1] / float(len(ref))


def parse_rttm(path: Path) -> list[DiarizationTurn]:
    turns: list[DiarizationTurn] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        turns.append(DiarizationTurn(start=start, end=start + duration, speaker=speaker))
    turns.sort(key=lambda t: (t.start, t.end, t.speaker))
    return turns


def speaker_overlap_matrix(
    predicted: list[DiarizationTurn],
    reference: list[DiarizationTurn],
) -> tuple[list[str], list[str], list[list[float]]]:
    pred_speakers = sorted({t.speaker for t in predicted})
    ref_speakers = sorted({t.speaker for t in reference})
    matrix = [[0.0 for _ in ref_speakers] for _ in pred_speakers]
    pred_index = {speaker: i for i, speaker in enumerate(pred_speakers)}
    ref_index = {speaker: i for i, speaker in enumerate(ref_speakers)}
    for p in predicted:
        pi = pred_index[p.speaker]
        for r in reference:
            ov = overlap_seconds(p.start, p.end, r.start, r.end)
            if ov > 0.0:
                matrix[pi][ref_index[r.speaker]] += ov
    return pred_speakers, ref_speakers, matrix


def best_speaker_mapping(
    pred_speakers: list[str],
    ref_speakers: list[str],
    matrix: list[list[float]],
) -> dict[str, str]:
    if not pred_speakers or not ref_speakers:
        return {}

    n_pred = len(pred_speakers)
    n_ref = len(ref_speakers)

    # Exact bitmask DP when feasible; otherwise greedy fallback.
    if max(n_pred, n_ref) <= 14:
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def solve(i: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
            if i >= n_pred:
                return 0.0, tuple()
            best_score, best_pairs = solve(i + 1, used_mask)
            for j in range(n_ref):
                if used_mask & (1 << j):
                    continue
                score_next, pairs_next = solve(i + 1, used_mask | (1 << j))
                score_here = score_next + matrix[i][j]
                if score_here > best_score:
                    best_score = score_here
                    best_pairs = pairs_next + ((i, j),)
            return best_score, best_pairs

        _, pairs = solve(0, 0)
        mapping = {pred_speakers[i]: ref_speakers[j] for i, j in pairs if matrix[i][j] > 0.0}
        return mapping

    mapping: dict[str, str] = {}
    used_refs: set[str] = set()
    for i, pred in enumerate(pred_speakers):
        ranked = sorted(
            ((matrix[i][j], ref_speakers[j]) for j in range(n_ref)),
            key=lambda x: x[0],
            reverse=True,
        )
        for score, ref in ranked:
            if score <= 0.0:
                break
            if ref not in used_refs:
                mapping[pred] = ref
                used_refs.add(ref)
                break
    return mapping


def compute_diarization_precision(
    predicted: list[DiarizationTurn],
    reference: list[DiarizationTurn],
) -> Optional[float]:
    if not predicted or not reference:
        return None
    pred_speakers, ref_speakers, matrix = speaker_overlap_matrix(predicted, reference)
    mapping = best_speaker_mapping(pred_speakers, ref_speakers, matrix)
    if not mapping:
        return 0.0

    true_positive = 0.0
    predicted_total = 0.0
    ref_by_speaker: dict[str, list[DiarizationTurn]] = {}
    for ref in reference:
        ref_by_speaker.setdefault(ref.speaker, []).append(ref)

    for pred in predicted:
        predicted_total += max(0.0, pred.end - pred.start)
        mapped_ref = mapping.get(pred.speaker)
        if not mapped_ref:
            continue
        for ref_turn in ref_by_speaker.get(mapped_ref, []):
            true_positive += overlap_seconds(pred.start, pred.end, ref_turn.start, ref_turn.end)

    if predicted_total <= 0.0:
        return None
    return min(1.0, true_positive / predicted_total)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

"""
Core pipeline orchestration for transcription and diarization.
- `timed_diarize_audio()`: Runs diarization and measures execution time.
- `run_pipeline_once()`: Executes the full pipeline (transcription + diarization + alignment) for a single audio file.
  - Handles parallelization (transcription on GPU, diarization on CPU).
  - Computes metrics (latency, WER, diarization precision).
  - Returns structured results (transcript, spans, turns, metrics, speaker aliases).
"""

#`timed_diarize_audio()`: Runs diarization and measures execution time.

def timed_diarize_audio(
    preprocessed: PreprocessedAudio,
    diarization_origin: str,
    device: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> tuple[list[DiarizationTurn], float]:
    started = time.perf_counter()
    turns = diarize_audio(
        preprocessed=preprocessed,
        diarization_origin=diarization_origin,
        device=device,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return turns, time.perf_counter() - started


#- `run_pipeline_once()`: Executes the full pipeline (transcription + diarization + alignment) for a single audio file.
##  - Handles parallelization (transcription on GPU, diarization on CPU).
##  - Computes metrics (latency, WER, diarization precision).
##  - Returns structured results (transcript, spans, turns, metrics, speaker aliases).

def run_pipeline_once(
    audio_path: Path,
    preprocessed: PreprocessedAudio,
    args: argparse.Namespace,
    hardware: HardwareProfile,
    whisper_model: str,
    diarization_origin: str,
    transcription_device: str,
    diarization_device: str,
    compute_type: str,
    beam_size: int,
    persisted_aliases: OrderedDict[str, str],
) -> tuple[RunResult, OrderedDict[str, str]]:
    start_total = time.perf_counter()
    metrics = RunMetrics(audio_duration_sec=preprocessed.duration_sec)

    transcription_result: list[ASRSegment] = []
    detected_language: Optional[str] = args.language
    turns: list[DiarizationTurn] = []

    can_parallelize = transcription_device == "cuda" and diarization_device == "cpu"

    if can_parallelize:
        with ThreadPoolExecutor(max_workers=2) as executor:
            diarization_future = executor.submit(
                timed_diarize_audio,
                preprocessed,
                diarization_origin,
                diarization_device,
                args.num_speakers,
                args.min_speakers,
                args.max_speakers,
            )
            transcription_started = time.perf_counter()
            transcription_result, detected_language = transcribe_windows(
                preprocessed=preprocessed,
                whisper_model=whisper_model,
                device=transcription_device,
                compute_type=compute_type,
                language=args.language,
                beam_size=beam_size,
                keep_intermediates=args.keep_intermediates,
                strict_local=not args.allow_online_model_resolution,
            )
            metrics.transcription_sec = time.perf_counter() - transcription_started
            turns, metrics.diarization_sec = diarization_future.result()
    else:
        transcription_started = time.perf_counter()
        transcription_result, detected_language = transcribe_windows(
            preprocessed=preprocessed,
            whisper_model=whisper_model,
            device=transcription_device,
            compute_type=compute_type,
            language=args.language,
            beam_size=beam_size,
            keep_intermediates=args.keep_intermediates,
            strict_local=not args.allow_online_model_resolution,
        )
        metrics.transcription_sec = time.perf_counter() - transcription_started

        turns, metrics.diarization_sec = timed_diarize_audio(
            preprocessed,
            diarization_origin,
            diarization_device,
            args.num_speakers,
            args.min_speakers,
            args.max_speakers,
        )

    if not transcription_result:
        raise RuntimeError("Whisper returned no speech segments after preprocessing.")
    if not turns:
        raise RuntimeError("Diarization returned no speaker turns.")

    alignment_started = time.perf_counter()
    spans = build_word_level_spans(transcription_result, turns, args.max_span_gap)
    if not spans:
        spans = build_segment_level_spans(transcription_result, turns)
    spans, alias_map = relabel_speakers(spans, persisted_aliases)
    spans = merge_adjacent_spans(spans, args.max_span_gap)
    metrics.alignment_sec = time.perf_counter() - alignment_started

    transcript = render_transcript(spans)
    transcript_hash = sha256_text(transcript)

    if args.reference_transcript:
        ref_text = extract_reference_transcript_text(Path(args.reference_transcript).expanduser().resolve())
        metrics.wer = compute_wer(ref_text, transcript)
    if args.reference_rttm:
        reference_turns = parse_rttm(Path(args.reference_rttm).expanduser().resolve())
        metrics.diarization_precision = compute_diarization_precision(turns, reference_turns)

    metrics.total_sec = time.perf_counter() - start_total
    metrics.real_time_factor = (
        metrics.total_sec / metrics.audio_duration_sec if metrics.audio_duration_sec > 0 else 0.0
    )

    return RunResult(
        transcript=transcript,
        spans=spans,
        turns=turns,
        metrics=metrics,
        detected_language=detected_language,
        transcript_sha256=transcript_hash,
        whisper_model=whisper_model,
        diarization_model=diarization_origin,
        transcription_device=transcription_device,
        diarization_device=diarization_device,
        compute_type=compute_type,
    ), alias_map


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

"""
Generates structured reports for pipeline runs, including CLEAR-style metrics.
- `build_report()`: Compiles a comprehensive JSON report with:
  - Input/output paths and runtime configuration.
  - Hardware profile and recommendations.
  - Latency, efficacy, and reliability metrics (WER, diarization precision, RTF).
  - Consistency checks (deterministic outputs, success/failure rates).
  - Sense-Think-Act-Observe breakdown for transparency.
"""

def build_report(
    audio_path: Path,
    output_path: Path,
    rttm_path: Path,
    state_db_path: Path,
    hardware: HardwareProfile,
    args: argparse.Namespace,
    whisper_model: str,
    diarization_origin: str,
    transcription_device: str,
    diarization_device: str,
    compute_type: str,
    beam_size: int,
    preprocessing_seconds: float,
    successful_results: list[RunResult],
    failures: list[str],
) -> dict[str, Any]:
    first = successful_results[0] if successful_results else None
    success_count = len(successful_results)
    requested_runs = max(1, int(args.consistency_runs))
    output_hashes = sorted({r.transcript_sha256 for r in successful_results})

    clear_section: dict[str, Any] = {
        "latency": None,
        "efficacy": None,
        "reliability": {
            "requested_runs": requested_runs,
            "successful_runs": success_count,
            "failed_runs": len(failures),
            "success_rate": (success_count / float(requested_runs)) if requested_runs else None,
            "unique_transcript_hashes": len(output_hashes),
            "deterministic_outputs": len(output_hashes) <= 1 if output_hashes else None,
            "failure_messages": failures,
        },
    }

    if first is not None:
        avg_total = sum(r.metrics.total_sec for r in successful_results) / float(success_count)
        avg_transcription = sum(r.metrics.transcription_sec for r in successful_results) / float(success_count)
        avg_diarization = sum(r.metrics.diarization_sec for r in successful_results) / float(success_count)
        avg_alignment = sum(r.metrics.alignment_sec for r in successful_results) / float(success_count)
        avg_rtf = sum(r.metrics.real_time_factor for r in successful_results) / float(success_count)
        clear_section["latency"] = {
            "audio_duration_sec": first.metrics.audio_duration_sec,
            "preprocessing_sec": preprocessing_seconds,
            "avg_transcription_sec": avg_transcription,
            "avg_diarization_sec": avg_diarization,
            "avg_alignment_sec": avg_alignment,
            "avg_total_sec": avg_total,
            "avg_real_time_factor": avg_rtf,
        }
        clear_section["efficacy"] = {
            "detected_language": first.detected_language,
            "wer": first.metrics.wer,
            "diarization_precision": first.metrics.diarization_precision,
            "speaker_count": len({span.speaker for span in first.spans}),
        }

    return {
        "generated_at_utc": utc_now_iso(),
        "input": str(audio_path),
        "output": str(output_path),
        "predicted_rttm": str(rttm_path),
        "state_db": str(state_db_path),
        "runtime": {
            "transcription_device": transcription_device,
            "diarization_device": diarization_device,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "whisper_model": whisper_model,
            "diarization_model": diarization_origin,
            "strict_local": not args.allow_online_model_resolution,
        },
        "hardware_profile": sanitize_for_json(hardware),
        "sense_think_act_observe": {
            "sense": {
                "gpu_name": hardware.gpu_name,
                "vram_gb": hardware.vram_gb,
                "recommended_model": hardware.recommended_whisper_model,
                "recommended_compute_type": hardware.recommended_compute_type,
                "recommended_diarization_device": hardware.recommended_diarization_device,
                "recommendation_reason": hardware.recommendation_reason,
            },
            "think": {
                "signal_enhancement": args.signal_enhancement,
                "vad_frame_ms": args.vad_frame_ms,
                "vad_hop_ms": args.vad_hop_ms,
                "vad_padding_ms": args.vad_padding_ms,
                "vad_threshold_scale": args.vad_threshold_scale,
                "max_chunk_sec": args.max_chunk_sec,
            },
            "act": {
                "pipeline_modules": ["preprocessing", "transcription", "diarization", "alignment", "persistence"],
                "parallel_diarization": transcription_device == "cuda" and diarization_device == "cpu",
            },
            "observe": clear_section,
        },
        "consistency": {
            "requested_runs": requested_runs,
            "successful_runs": success_count,
            "transcript_hashes": output_hashes,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.exists():
        error(f"Input file not found: {audio_path}")
        return 1

    configure_offline_mode(args.allow_online_model_resolution)

    output_path, report_path, state_db_path = resolve_output_paths(audio_path, args)
    diarization_origin = resolve_diarization_origin(args.diarization_config)

    
    hardware = probe_hardware(
        transcription_device=args.transcription_device,
        diarization_device=args.diarization_device,
        model_profile=args.model_profile,
        beam_size_override=args.beam_size,
    )
    maybe_warn_on_cuda13_stack(hardware)

    whisper_model = choose_whisper_model(args, hardware)
    transcription_device = hardware.device
    diarization_device = choose_diarization_device(args, hardware)
    compute_type = choose_compute_type(args, hardware)
    beam_size = choose_beam_size(args, hardware)

    validate_model_choice(
        whisper_model=whisper_model,
        compute_type=compute_type,
        hardware=hardware,
        force_risky_model=args.force_risky_model,
    )

    log(f"Input: {audio_path}")
    log(f"Whisper model: {whisper_model}")
    log(f"Transcription device: {transcription_device}")
    log(f"Diarization device: {diarization_device}")
    log(f"Compute type: {compute_type}")
    log(f"Beam size: {beam_size}")
    if hardware.gpu_name:
        log(f"GPU: {hardware.gpu_name} ({hardware.vram_gb} GB VRAM, torch CUDA {hardware.torch_cuda_version})")
    log(f"Recommended config: {hardware.recommended_whisper_model} / {hardware.recommended_compute_type} / diarization on {hardware.recommended_diarization_device}")

    conn = init_state_db(state_db_path)
    audio_sha = sha256_file(audio_path)
    persisted_aliases = load_alias_map(conn, audio_sha)

    run_group_id = str(uuid.uuid4())
    successful_results: list[RunResult] = []
    failures: list[str] = []
    chosen_alias_map: Optional[OrderedDict[str, str]] = None

    workdir_ctx: Optional[tempfile.TemporaryDirectory] = None
    if args.keep_intermediates:
        temp_dir = output_path.with_name(f"{audio_path.stem}_intermediates")
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        workdir_ctx = tempfile.TemporaryDirectory(prefix="local_asr_sd_")
        temp_dir = Path(workdir_ctx.name)

    try:
        preprocessing_started = time.perf_counter()
        preprocessed = preprocess_audio(audio_path, temp_dir, args)
        preprocessing_seconds = time.perf_counter() - preprocessing_started

        for run_index in range(1, max(1, int(args.consistency_runs)) + 1):
            run_id = str(uuid.uuid4())
            record_run_start(
                conn=conn,
                run_id=run_id,
                run_group_id=run_group_id,
                run_index=run_index,
                audio_path=audio_path,
                audio_sha256=audio_sha,
                whisper_model=whisper_model,
                diarization_model=diarization_origin,
                transcription_device=transcription_device,
                diarization_device=diarization_device,
                compute_type=compute_type,
                output_path=output_path,
                report_path=report_path,
                args=args,
            )

            try:
                log(f"Processing... Run {run_index}/{max(1, int(args.consistency_runs))}...")
                result, alias_map = run_pipeline_once(
                    audio_path=audio_path,
                    preprocessed=preprocessed,
                    args=args,
                    hardware=hardware,
                    whisper_model=whisper_model,
                    diarization_origin=diarization_origin,
                    transcription_device=transcription_device,
                    diarization_device=diarization_device,
                    compute_type=compute_type,
                    beam_size=beam_size,
                    persisted_aliases=persisted_aliases,
                )
                result.metrics.preprocessing_sec = preprocessing_seconds
                successful_results.append(result)
                chosen_alias_map = alias_map
                record_run_success(conn, run_id, result)
                persist_utterances(conn, run_id, result.spans)
                log(
                    f"Run {run_index} complete: time={result.metrics.total_sec:.2f}s, "
                    f"speakers={len({s.speaker for s in result.spans})}"
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                failures.append(f"Run {run_index}: {type(exc).__name__}: {exc}")
                record_run_failure(conn, run_id, exc)
                error(f"Run {run_index} failed: {exc}")
                traceback.print_exc()
            finally:
                cleanup_cuda_cache()
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()

    if not successful_results:
        error("All runs failed. See state store and stderr for details.")
        conn.close()
        return 1

    if chosen_alias_map is not None:
        persist_alias_map(conn, audio_sha, chosen_alias_map)

    best_result = successful_results[0]
    output_path.write_text(best_result.transcript, encoding="utf-8")
    rttm_path = output_path.with_suffix(".rttm")
    write_rttm(best_result.turns, rttm_path, audio_path.stem)

    report = build_report(
        audio_path=audio_path,
        output_path=output_path,
        rttm_path=rttm_path,
        state_db_path=state_db_path,
        hardware=hardware,
        args=args,
        whisper_model=whisper_model,
        diarization_origin=diarization_origin,
        transcription_device=transcription_device,
        diarization_device=diarization_device,
        compute_type=compute_type,
        beam_size=beam_size,
        preprocessing_seconds=successful_results[0].metrics.preprocessing_sec,
        successful_results=successful_results,
        failures=failures,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    conn.close()

    log(f"Transcript saved: {output_path}")
    log(f"RTTM saved: {rttm_path}")
    log(f"Report saved: {report_path}")
    log(f"State store saved: {state_db_path}")
    if best_result.detected_language:
        log(f"Detected language: {best_result.detected_language}")
    if best_result.metrics.wer is not None:
        log(f"WER: {best_result.metrics.wer:.4f}")
    if best_result.metrics.diarization_precision is not None:
        log(f"Approx. diarization precision: {best_result.metrics.diarization_precision:.4f}")
    log(
        f"Latency: {best_result.metrics.total_sec:.2f}s (RTF={best_result.metrics.real_time_factor:.3f} for {best_result.metrics.audio_duration_sec:.2f}s audio)"
    )
    if args.consistency_runs > 1:
        log(
            f"Consistency summary: {len(successful_results)}/{args.consistency_runs} successful runs, "
            f"unique transcript hashes={len({r.transcript_sha256 for r in successful_results})}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
