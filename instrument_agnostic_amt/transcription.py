from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Literal

import torch
import torchaudio.functional as audio_functional

from .data.pitch_aliases import DEFAULT_DRUM_PITCH_ALIASES, parse_pitch_aliases
from .inference.audio import load_audio
from .inference.instruments import (
    filter_supported_instrument_class_ids,
    instrument_class_ids_from_names,
    normalize_instrument_class_name,
)
from .inference.midi import build_midi
from .inference.settings import resolve_inference_settings
from .inference.types import InferenceSettings, PredictedNote
from .inference.windowed import decode_notes
from .modeling.checkpoints import (
    CheckpointLoadReport,
    coerce_model_config,
    extract_inference_args,
    extract_model_config,
    load_checkpoint,
    load_compatible_weights,
    require_complete_inference_head,
)
from .modeling.model import AudioSemiCRFTransformer, SemiCRFModelConfig
from .runtime import (
    is_amp_supported,
    maybe_compile_forward,
    resolve_amp_dtype,
    resolve_device,
)
from .taxonomy.instrument_classes import INSTRUMENT_CLASSES


class TranscriberBusyError(RuntimeError):
    """Raised when one Transcriber is called concurrently."""


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """Decoded PCM audio paired with its source sample rate."""

    samples: torch.Tensor
    sample_rate: int

    def __post_init__(self) -> None:
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        if not isinstance(self.samples, torch.Tensor):
            raise TypeError("samples must be a torch.Tensor")
        if self.samples.device.type != "cpu":
            raise ValueError("samples must be on CPU")
        if self.samples.ndim != 2:
            raise ValueError("samples must have shape [channels, audio_frames]")
        if int(self.samples.shape[0]) not in (1, 2):
            raise ValueError("samples must have one or two channels")
        if int(self.samples.shape[1]) == 0:
            raise ValueError("samples must not be empty")
        if self.samples.dtype != torch.float32:
            raise ValueError("samples must have dtype float32")


@dataclass(frozen=True, slots=True)
class TranscriptionOptions:
    """AMT decoding options for one transcription call."""

    instrument: str | None = None
    allowed_instruments: tuple[str, ...] | None = None
    window_ms: int | None = None
    stride_ms: int | None = None
    window_batch_size: int = 1
    semi_crf_track_batch_size: int | None = None
    semi_crf_backend: Literal["torch", "triton"] = "torch"
    semi_crf_sparse_decode: bool = False
    semi_crf_sparse_topk_per_start: int = 16
    semi_crf_sparse_score_threshold: float | None = None
    semi_crf_sparse_max_span_ms: float | None = None
    instrument_pair_infer_topk: int = 256
    instrument_pair_gate_threshold: float = -3.0
    instrument_pair_max_pairs: int = 512
    merge_gap_ms: float | None = None
    merge_onset_ms: float = 50.0
    silence_gate_rms_dbfs: float | None = -72.0
    note_bias: float = 0.0
    use_boundary_head: bool = True
    midi_velocity: int = 100
    show_progress: bool = False

    def __post_init__(self) -> None:
        if self.instrument is not None and self.allowed_instruments is not None:
            raise ValueError(
                "instrument and allowed_instruments are mutually exclusive"
            )
        if self.allowed_instruments is not None and not self.allowed_instruments:
            raise ValueError("allowed_instruments must not be empty")
        if self.window_ms is not None and self.window_ms <= 0:
            raise ValueError("window_ms must be positive")
        if self.stride_ms is not None and self.stride_ms <= 0:
            raise ValueError("stride_ms must be positive")
        if self.window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive")
        if (
            self.semi_crf_track_batch_size is not None
            and self.semi_crf_track_batch_size <= 0
        ):
            raise ValueError("semi_crf_track_batch_size must be positive")
        if self.semi_crf_backend not in {"torch", "triton"}:
            raise ValueError("semi_crf_backend must be 'torch' or 'triton'")
        if self.semi_crf_sparse_topk_per_start <= 0:
            raise ValueError("semi_crf_sparse_topk_per_start must be positive")
        if (
            self.semi_crf_sparse_max_span_ms is not None
            and self.semi_crf_sparse_max_span_ms <= 0
        ):
            raise ValueError("semi_crf_sparse_max_span_ms must be positive")
        if self.semi_crf_sparse_decode and self.semi_crf_sparse_max_span_ms is None:
            raise ValueError(
                "semi_crf_sparse_decode requires semi_crf_sparse_max_span_ms"
            )
        if self.semi_crf_sparse_decode and self.semi_crf_backend != "torch":
            raise ValueError("semi_crf_sparse_decode requires semi_crf_backend='torch'")
        if self.instrument_pair_infer_topk < 0:
            raise ValueError("instrument_pair_infer_topk must be non-negative")
        if self.instrument_pair_max_pairs <= 0:
            raise ValueError("instrument_pair_max_pairs must be positive")
        if self.merge_gap_ms is not None and self.merge_gap_ms < 0:
            raise ValueError("merge_gap_ms must be non-negative")
        if self.merge_onset_ms < 0:
            raise ValueError("merge_onset_ms must be non-negative")
        if (
            self.silence_gate_rms_dbfs is not None
            and self.silence_gate_rms_dbfs > 0
        ):
            raise ValueError("silence_gate_rms_dbfs must be <= 0")
        if not 1 <= self.midi_velocity <= 127:
            raise ValueError("midi_velocity must be within MIDI 1..127")


@dataclass(frozen=True, slots=True)
class MidiExportOptions:
    """Options applied while converting predicted notes to MIDI."""

    min_midi_note_ms: float = 5.0
    max_midi_melodic_instruments: int = 15
    instrument_volumes: Mapping[str, int] | None = None
    drum_pitch_aliases: Mapping[int, int] | None = field(
        default_factory=lambda: dict(DEFAULT_DRUM_PITCH_ALIASES)
    )

    def __post_init__(self) -> None:
        if self.min_midi_note_ms < 0:
            raise ValueError("min_midi_note_ms must be non-negative")
        if self.max_midi_melodic_instruments < 0:
            raise ValueError("max_midi_melodic_instruments must be non-negative")

        if self.instrument_volumes is not None:
            if self.instrument_volumes:
                instrument_class_ids_from_names(self.instrument_volumes)
            normalized_volumes: dict[str, int] = {}
            for name, raw_volume in self.instrument_volumes.items():
                volume = int(raw_volume)
                if not 0 <= volume <= 127:
                    raise ValueError(
                        f"Volume for instrument '{name}' must be within MIDI 0..127"
                    )
                normalized_volumes[normalize_instrument_class_name(name)] = volume
            object.__setattr__(
                self,
                "instrument_volumes",
                normalized_volumes,
            )

        if self.drum_pitch_aliases is not None:
            object.__setattr__(
                self,
                "drum_pitch_aliases",
                parse_pitch_aliases(self.drum_pitch_aliases),
            )


@dataclass(frozen=True, slots=True)
class TranscriptionModelInfo:
    """Checkpoint identity and effective runtime settings."""

    checkpoint_path: Path
    sample_rate: int
    input_audio_channels: int
    num_instrument_classes: int
    supported_instruments: tuple[str, ...]
    semi_crf_version: str
    device: str
    amp_enabled: bool
    amp_dtype: str
    compile_enabled: bool
    compile_mode: str | None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """In-memory transcription output.

    ``notes`` contains decoder output before MIDI export applies pitch aliases,
    track remapping, overlap truncation, and minimum-duration adjustments.
    ``midi_bytes`` is the authoritative exported MIDI representation.
    """

    notes: tuple[PredictedNote, ...]
    midi_bytes: bytes
    inference_stats: Mapping[str, int]
    midi_stats: Mapping[str, int]
    settings: InferenceSettings
    sample_rate: int
    model_info: TranscriptionModelInfo


@dataclass(frozen=True, slots=True)
class _LoadedCheckpoint:
    model: AudioSemiCRFTransformer
    config: SemiCRFModelConfig
    checkpoint_args: Mapping[str, object]
    report: CheckpointLoadReport


def _load_checkpoint_bundle(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> _LoadedCheckpoint:
    checkpoint = load_checkpoint(checkpoint_path)
    config = coerce_model_config(
        extract_model_config(checkpoint),
        for_inference=True,
    )
    if int(config.num_instrument_classes) > len(INSTRUMENT_CLASSES):
        raise ValueError(
            "Checkpoint instrument head has "
            f"{int(config.num_instrument_classes)} classes, but the bundled "
            f"taxonomy defines {len(INSTRUMENT_CLASSES)}"
        )
    model = AudioSemiCRFTransformer(config)
    report = load_compatible_weights(
        model,
        checkpoint,
        prefer_ema=True,
        require_complete_backbone=True,
    )
    require_complete_inference_head(model, report)
    model.to(device)
    model.eval()
    return _LoadedCheckpoint(
        model=model,
        config=config,
        checkpoint_args=extract_inference_args(checkpoint),
        report=report,
    )


class Transcriber:
    """Long-lived transcription handle for one model checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        model: AudioSemiCRFTransformer,
        config: SemiCRFModelConfig,
        checkpoint_args: Mapping[str, object],
        load_report: CheckpointLoadReport,
        device: torch.device,
        amp_enabled: bool,
        amp_dtype: torch.dtype,
        inference_model: torch.nn.Module,
        compile_enabled: bool,
        compile_mode: str | None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._model = model
        self._config = config
        self._checkpoint_args = dict(checkpoint_args)
        self._load_report = load_report
        self._device = device
        self._amp_enabled = amp_enabled
        self._amp_dtype = amp_dtype
        self._inference_model = inference_model
        self._call_lock = Lock()
        self._model_info = TranscriptionModelInfo(
            checkpoint_path=checkpoint_path,
            sample_rate=int(config.sample_rate),
            input_audio_channels=int(config.input_audio_channels),
            num_instrument_classes=int(config.num_instrument_classes),
            supported_instruments=tuple(
                INSTRUMENT_CLASSES[: int(config.num_instrument_classes)]
            ),
            semi_crf_version=str(config.semi_crf_version),
            device=str(device),
            amp_enabled=bool(amp_enabled),
            amp_dtype=str(amp_dtype),
            compile_enabled=bool(compile_enabled),
            compile_mode=compile_mode,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = "auto",
        amp: bool = False,
        amp_dtype: Literal["fp16", "bf16"] | None = None,
        compile: bool = False,
        compile_mode: str = "default",
    ) -> Transcriber:
        """Load one trusted local checkpoint without downloading files."""

        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint is not a file: {resolved_checkpoint}")
        target_device = resolve_device(device)
        bundle = _load_checkpoint_bundle(
            resolved_checkpoint,
            device=target_device,
        )
        inference_model = maybe_compile_forward(
            bundle.model,
            enabled=bool(compile),
            mode=str(compile_mode),
        )
        resolved_amp_dtype = resolve_amp_dtype(target_device, amp_dtype)
        return cls(
            checkpoint_path=resolved_checkpoint,
            model=bundle.model,
            config=bundle.config,
            checkpoint_args=bundle.checkpoint_args,
            load_report=bundle.report,
            device=target_device,
            amp_enabled=bool(amp and is_amp_supported(target_device)),
            amp_dtype=resolved_amp_dtype,
            inference_model=inference_model,
            compile_enabled=bool(compile),
            compile_mode=str(compile_mode) if compile else None,
        )

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def sample_rate(self) -> int:
        return int(self._config.sample_rate)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def supported_instruments(self) -> tuple[str, ...]:
        return self._model_info.supported_instruments

    @property
    def model_info(self) -> TranscriptionModelInfo:
        return self._model_info

    @property
    def load_report(self) -> CheckpointLoadReport:
        return self._load_report

    def transcribe(
        self,
        audio: str | Path | DecodedAudio,
        *,
        options: TranscriptionOptions | None = None,
        midi_options: MidiExportOptions | None = None,
    ) -> TranscriptionResult:
        """Transcribe one path or decoded PCM buffer into an in-memory result."""

        if not self._call_lock.acquire(blocking=False):
            raise TranscriberBusyError(
                "This Transcriber is already processing another call"
            )
        try:
            return self._transcribe(audio, options=options, midi_options=midi_options)
        finally:
            self._call_lock.release()

    def _transcribe(
        self,
        audio: str | Path | DecodedAudio,
        *,
        options: TranscriptionOptions | None,
        midi_options: MidiExportOptions | None,
    ) -> TranscriptionResult:

        call_options = options or TranscriptionOptions()
        export_options = midi_options or MidiExportOptions()
        instrument_id, allowed_instrument_ids = self._resolve_instruments(call_options)
        settings = resolve_inference_settings(
            self._config,
            self._checkpoint_args,
            call_options,
            allowed_instrument_ids=allowed_instrument_ids,
        )
        if settings.semi_crf_backend == "triton" and self._device.type != "cuda":
            raise ValueError("semi_crf_backend='triton' requires a CUDA device")
        waveform = self._prepare_audio(audio)
        notes, inference_stats = decode_notes(
            self._model,
            self._config,
            waveform,
            instrument_filter_id=instrument_id,
            device=self._device,
            amp_enabled=self._amp_enabled,
            amp_dtype=self._amp_dtype,
            settings=settings,
            velocity=int(call_options.midi_velocity),
            forward_model=self._inference_model,
        )
        midi, midi_stats = build_midi(
            notes,
            sample_rate=self.sample_rate,
            instrument_id=instrument_id,
            min_midi_note_ms=float(export_options.min_midi_note_ms),
            max_midi_melodic_instruments=int(
                export_options.max_midi_melodic_instruments
            ),
            instrument_volumes=(
                None
                if export_options.instrument_volumes is None
                else dict(export_options.instrument_volumes)
            ),
            drum_pitch_aliases=export_options.drum_pitch_aliases,
            return_stats=True,
        )
        midi_buffer = io.BytesIO()
        midi.write(midi_buffer)
        return TranscriptionResult(
            notes=tuple(notes),
            midi_bytes=midi_buffer.getvalue(),
            inference_stats=dict(inference_stats),
            midi_stats=dict(midi_stats),
            settings=settings,
            sample_rate=self.sample_rate,
            model_info=self._model_info,
        )

    def _prepare_audio(self, audio: str | Path | DecodedAudio) -> torch.Tensor:
        if not isinstance(audio, DecodedAudio):
            return load_audio(Path(audio), target_sample_rate=self.sample_rate)

        waveform = audio.samples.detach()
        if int(waveform.shape[0]) == 1:
            waveform = waveform.repeat(2, 1)
        if int(audio.sample_rate) != self.sample_rate:
            waveform = audio_functional.resample(
                waveform,
                int(audio.sample_rate),
                self.sample_rate,
            )
        return waveform.contiguous()

    def _resolve_instruments(
        self,
        options: TranscriptionOptions,
    ) -> tuple[int | None, tuple[int, ...] | None]:
        instrument_id = None
        requested_ids = None
        if options.instrument is not None:
            requested_ids = instrument_class_ids_from_names((options.instrument,))
            instrument_id = int(requested_ids[0])
        elif options.allowed_instruments is not None:
            requested_ids = instrument_class_ids_from_names(
                options.allowed_instruments
            )
        allowed_ids = filter_supported_instrument_class_ids(
            requested_ids,
            num_model_classes=int(self._config.num_instrument_classes),
        )
        return instrument_id, allowed_ids
