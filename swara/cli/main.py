"""Command Line Interface (CLI) for Swara."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional
import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from swara import __version__
from swara.core.audio import load_audio, record_microphone, save_audio
from swara.core.config import settings
from swara.engine import Swara
from swara.models.manager import ModelManager
from swara.models.registry import MODEL_REGISTRY

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="Swara")
def cli() -> None:
    """Swara: Ultra-low-latency, lightweight local speaker intelligence AI.

    CSKY - 'Every Voice Has an Identity.'
    """
    pass


@cli.command("doctor")
def doctor_cmd() -> None:
    """Diagnoses system environment, dependencies, models, and hardware capabilities."""
    console.print(Panel.fit("[bold cyan]Swara Environment & System Diagnostics[/bold cyan]", border_style="cyan"))

    table = Table(title="System & Runtime Checks", header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    # 1. Python Check
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 9)
    table.add_row("Python Version", "[green]PASS[/green]" if py_ok else "[red]FAIL[/red]", f"v{py_ver} ({sys.executable})")

    # 2. Hardware: CPU & RAM
    import psutil
    cpu_count = psutil.cpu_count(logical=True)
    mem_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    avail_gb = round(psutil.virtual_memory().available / (1024**3), 2)
    table.add_row("CPU Logical Cores", "[green]PASS[/green]", f"{cpu_count} threads available")
    table.add_row("RAM Memory", "[green]PASS[/green]", f"{mem_gb} GB total, {avail_gb} GB free")

    # 3. Hardware: GPU / Accelerator
    import onnxruntime as ort
    providers = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in providers
    table.add_row("ONNX Providers", "[green]OK[/green]", ", ".join(providers))
    table.add_row("GPU Acceleration", "[green]CUDA Ready[/green]" if has_cuda else "[yellow]CPU Only[/yellow]", "CUDA available" if has_cuda else "Running on optimized CPU")

    # 4. Audio Driver / Microphone
    mic_ok = False
    mic_detail = "sounddevice unavailable"
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if input_devs:
            mic_ok = True
            mic_detail = f"{len(input_devs)} input device(s) found"
        else:
            mic_detail = "No input audio devices found"
    except Exception as e:
        mic_detail = f"Microphone check error: {e}"
    table.add_row("Microphone Support", "[green]PASS[/green]" if mic_ok else "[yellow]NO INPUT[/yellow]", mic_detail)

    # 5. Model Artifacts
    mgr = ModelManager()
    all_models_ready = True
    for mid, spec in MODEL_REGISTRY.items():
        if spec.is_default:
            is_down = mgr.is_downloaded(mid)
            is_valid = mgr.verify_checksum(mid) if is_down else False
            if not (is_down and is_valid):
                all_models_ready = False
            status_str = "[green]VERIFIED[/green]" if is_valid else ("[yellow]CORRUPT[/yellow]" if is_down else "[red]MISSING[/red]")
            table.add_row(f"Model: {mid}", status_str, f"{spec.filename} ({spec.size_mb:.2f} MB)")

    console.print(table)

    if all_models_ready:
        console.print("[bold green]All systems operational! Swara is ready for offline inference.[/bold green]")
    else:
        console.print("[bold yellow]Action required: run 'swara models download' to fetch required models.[/bold yellow]")


@cli.group("models")
def models_grp() -> None:
    """Manages model artifacts, downloads, and checksum verifications."""
    pass


@models_grp.command("list")
def models_list_cmd() -> None:
    """Lists registered models and local cache status."""
    mgr = ModelManager()
    items = mgr.list_models()

    table = Table(title="Swara Model Artifacts Registry", header_style="bold blue")
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Architecture", style="white")
    table.add_column("Size", justify="right")
    table.add_column("License", style="yellow")
    table.add_column("Downloaded", style="bold")
    table.add_column("Checksum", style="bold")

    for it in items:
        down_str = "[green]YES[/green]" if it["is_downloaded"] else "[red]NO[/red]"
        check_str = "[green]PASS[/green]" if it["is_verified"] else ("[yellow]FAIL[/yellow]" if it["is_downloaded"] else "-")
        table.add_row(
            it["model_id"],
            it["architecture"],
            f"{it['size_mb']} MB",
            it["license"],
            down_str,
            check_str,
        )

    console.print(table)


@models_grp.command("download")
@click.option("--all", "include_all", is_flag=True, help="Download all models including optional INT8 variants.")
@click.option("--force", is_flag=True, help="Force re-download even if already present.")
def models_download_cmd(include_all: bool, force: bool) -> None:
    """Downloads approved model artifacts with SHA-256 verification."""
    mgr = ModelManager()
    console.print("[bold cyan]Fetching model artifacts from official repositories...[/bold cyan]")
    for mid, spec in MODEL_REGISTRY.items():
        if spec.is_default or include_all:
            with console.status(f"Downloading {mid} ({spec.size_mb:.2f} MB)..."):
                path = mgr.download_model(mid, force=force)
            console.print(f"[green]Downloaded and verified[/green] {mid} -> {path}")
    console.print("[bold green]Model acquisition complete.[/bold green]")


@cli.command("enroll")
@click.argument("speaker_id")
@click.option("--audio", "audio_path", required=True, type=click.Path(exists=True), help="Path to audio file (WAV/FLAC).")
@click.option("--name", default=None, help="Optional human-readable display name.")
def enroll_cmd(speaker_id: str, audio_path: str, name: Optional[str]) -> None:
    """Enrolls a voice sample for a speaker identity."""
    engine = Swara()
    with console.status(f"Processing and enrolling '{speaker_id}'..."):
        profile = engine.enroll(speaker_id=speaker_id, audio_path=audio_path, name=name)

    table = Table(title=f"Enrolled Speaker Profile: {speaker_id}", header_style="bold green")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Speaker ID", profile.speaker_id)
    table.add_row("Display Name", profile.name or "-")
    table.add_row("Enrolled Samples", str(profile.sample_count))
    table.add_row("Total Audio", f"{profile.total_audio_duration_s:.2f} seconds")
    table.add_row("Embedding Dim", str(profile.embedding_dim))
    table.add_row("Model Backbone", profile.model_version)
    console.print(table)
    console.print(f"[bold green]Successfully enrolled '{speaker_id}' into local database.[/bold green]")


@cli.command("identify")
@click.option("--audio", "audio_path", required=True, type=click.Path(exists=True), help="Path to probe audio file.")
@click.option("--threshold", default=None, type=float, help="Custom cosine similarity threshold.")
def identify_cmd(audio_path: str, threshold: Optional[float]) -> None:
    """Identifies speaker from an audio sample against enrolled voices."""
    engine = Swara()
    with console.status("Extracting embedding and searching enrolled database..."):
        res = engine.identify(audio_path=audio_path, threshold=threshold)

    table = Table(title="Speaker Identification Result", header_style="bold cyan")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")

    if res.is_match:
        table.add_row("Identity Result", f"[bold green]{res.speaker_id}[/bold green]")
        table.add_row("Match Status", "[bold green]VERIFIED MATCH[/bold green]")
        table.add_row("Cosine Similarity", f"{res.similarity:.4f}")
        table.add_row("Calibrated Confidence", f"{res.confidence * 100:.1f}%")
    else:
        table.add_row("Identity Result", "[bold red]UNKNOWN SPEAKER[/bold red]")
        table.add_row("Match Status", "[yellow]REJECTED (Insufficient evidence)[/yellow]")
        if res.similarity is not None:
            table.add_row("Best Candidate Sim", f"{res.similarity:.4f}")

    table.add_row("Audio Duration", f"{res.audio_duration:.2f}s")
    console.print(table)

    if res.candidates:
        c_table = Table(title="Top Candidate Matches", header_style="bold magenta")
        c_table.add_column("Rank", justify="right")
        c_table.add_column("Speaker ID", style="cyan")
        c_table.add_column("Similarity", justify="right")
        c_table.add_column("Confidence", justify="right")
        c_table.add_column("Match?", justify="center")

        for idx, c in enumerate(res.candidates, 1):
            m_str = "[green]YES[/green]" if c.is_match else "[red]NO[/red]"
            c_table.add_row(str(idx), c.speaker_id, f"{c.similarity:.4f}", f"{c.confidence*100:.1f}%", m_str)
        console.print(c_table)


@cli.command("verify")
@click.argument("speaker_id")
@click.option("--audio", "audio_path", required=True, type=click.Path(exists=True), help="Path to probe audio file.")
@click.option("--threshold", default=None, type=float, help="Custom threshold.")
def verify_cmd(speaker_id: str, audio_path: str, threshold: Optional[float]) -> None:
    """Performs 1:1 voice verification against an enrolled speaker."""
    engine = Swara()
    try:
        with console.status(f"Verifying against '{speaker_id}'..."):
            is_ver, sim, conf = engine.verify(speaker_id=speaker_id, audio_path=audio_path, threshold=threshold)
        status_text = "[bold green]VERIFIED[/bold green]" if is_ver else "[bold red]REJECTED[/bold red]"
        console.print(f"Target: [cyan]{speaker_id}[/cyan] | Result: {status_text} | Similarity: [bold]{sim:.4f}[/bold] | Confidence: [bold]{conf*100:.1f}%[/bold]")
    except KeyError:
        console.print(f"[bold red]Error: Speaker '{speaker_id}' is not enrolled.[/bold red]")


@cli.command("diarize")
@click.option("--audio", "audio_path", required=True, type=click.Path(exists=True), help="Path to conversation audio.")
@click.option("--threshold", default=None, type=float, help="Clustering threshold.")
@click.option("--min-speakers", default=None, type=int, help="Minimum speaker count hint.")
@click.option("--max-speakers", default=None, type=int, help="Maximum speaker count hint.")
def diarize_cmd(audio_path: str, threshold: Optional[float], min_speakers: Optional[int], max_speakers: Optional[int]) -> None:
    """Diarizes multi-speaker audio into time-aligned speaker segments."""
    engine = Swara()
    with console.status("Running VAD, embedding extraction, and clustering..."):
        res = engine.diarize(
            audio_path=audio_path,
            threshold=threshold,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            enhance_with_identity=True,
        )

    table = Table(title=f"Diarization Result ({res.num_speakers} unique speakers, {res.duration:.2f}s audio)", header_style="bold blue")
    table.add_column("Start (s)", justify="right")
    table.add_column("End (s)", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Speaker Cluster", style="cyan")
    table.add_column("Enrolled Identity", style="bold green")
    table.add_column("Match Score", justify="right")

    for seg in res.segments:
        ident_str = seg.identity or "[dim]unregistered[/dim]"
        score_str = f"{seg.similarity:.4f}" if seg.similarity is not None else "-"
        table.add_row(
            f"{seg.start:.2f}",
            f"{seg.end:.2f}",
            f"{seg.duration:.2f}s",
            seg.speaker_id,
            ident_str,
            score_str,
        )

    console.print(table)


@cli.group("speakers")
def speakers_grp() -> None:
    """Manages enrolled speaker profiles."""
    pass


@speakers_grp.command("list")
def speakers_list_cmd() -> None:
    """Lists all enrolled voice profiles."""
    engine = Swara()
    speakers = engine.list_speakers()

    if not speakers:
        console.print("[yellow]No speakers currently enrolled. Use 'swara enroll' to register a voice.[/yellow]")
        return

    table = Table(title="Enrolled Speakers Database", header_style="bold cyan")
    table.add_column("Speaker ID", style="cyan")
    table.add_column("Display Name", style="white")
    table.add_column("Samples", justify="right")
    table.add_column("Audio (s)", justify="right")
    table.add_column("Enrolled At", style="dim")

    for s in speakers:
        table.add_row(
            s.speaker_id,
            s.name or s.speaker_id,
            str(s.sample_count),
            f"{s.total_audio_duration_s:.1f}s",
            s.created_at[:19].replace("T", " "),
        )
    console.print(table)


@speakers_grp.command("delete")
@click.argument("speaker_id")
def speakers_delete_cmd(speaker_id: str) -> None:
    """Deletes an enrolled speaker profile."""
    engine = Swara()
    success = engine.delete_speaker(speaker_id)
    if success:
        console.print(f"[green]Successfully deleted speaker '{speaker_id}'.[/green]")
    else:
        console.print(f"[red]Speaker '{speaker_id}' not found.[/red]")


@cli.command("record")
@click.option("--output", "-o", default="samples/recorded_voice.wav", help="Output file path.")
@click.option("--duration", "-d", default=5.0, type=float, help="Recording duration in seconds.")
def record_cmd(output: str, duration: float) -> None:
    """Records audio from your microphone."""
    console.print(f"[bold cyan]Recording {duration:.1f} seconds of audio from microphone...[/bold cyan]")
    console.print("[yellow]Speak clearly into your microphone now...[/yellow]")
    chunk = record_microphone(duration_seconds=duration)
    save_audio(output, chunk)
    console.print(f"[bold green]Saved recording to {output} ({chunk.duration:.2f}s at {chunk.sample_rate}Hz)[/bold green]")


@cli.command("benchmark")
@click.option("--iterations", "-n", default=20, type=int, help="Number of benchmark iterations.")
def benchmark_cmd(iterations: int) -> None:
    """Runs latency, memory, and Real-Time Factor (RTF) benchmarks on local CPU."""
    console.print(Panel.fit("[bold cyan]Swara Production Inference Benchmark[/bold cyan]", border_style="cyan"))

    import psutil

    process = psutil.Process()

    # Measure cold start
    t0 = time.perf_counter()
    engine = Swara()
    cold_start_ms = (time.perf_counter() - t0) * 1000

    # Test audio: 2.0s utterance
    sr = 16000
    dur_s = 2.0
    num_samples = int(dur_s * sr)
    test_audio = (np.sin(2 * np.pi * 200 * np.linspace(0, dur_s, num_samples, dtype=np.float32))).astype(np.float32)

    # Warm-up
    for _ in range(3):
        engine.extract_embedding(test_audio)

    # Benchmark embeddings
    latencies = []
    for _ in range(iterations):
        t_start = time.perf_counter()
        engine.extract_embedding(test_audio)
        latencies.append((time.perf_counter() - t_start) * 1000)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg_latency = float(np.mean(latencies))
    rtf = (avg_latency / 1000.0) / dur_s

    # Memory
    mem_mb = process.memory_info().rss / (1024 * 1024)

    # Model sizes
    mgr = ModelManager()
    total_model_size_mb = sum(
        spec.size_mb for mid, spec in MODEL_REGISTRY.items() if spec.is_default
    )

    table = Table(title="Benchmark Results", header_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Measured Value", style="bold white")
    table.add_column("Engineering Target", style="dim")

    table.add_row("Cold Start Initialization", f"{cold_start_ms:.1f} ms", "< 500 ms")
    table.add_row("Warm Latency (P50)", f"{p50:.2f} ms", "< 30 ms")
    table.add_row("Warm Latency (P95)", f"{p95:.2f} ms", "< 50 ms")
    table.add_row("Warm Latency (P99)", f"{p99:.2f} ms", "< 80 ms")
    table.add_row("Real-Time Factor (RTF)", f"{rtf:.4f}", "< 0.10 (CPU)")
    table.add_row("Inference Speedup", f"{1.0/rtf:.1f}x real-time", "> 10x")
    table.add_row("Resident Memory (RSS)", f"{mem_mb:.1f} MB", "< 150 MB")
    table.add_row("Combined Model Artifacts", f"{total_model_size_mb:.2f} MB", "< 75 MB")

    console.print(table)


@cli.command("evaluate")
@click.option("--config", "-c", default="configs/evaluation.yaml", help="Path to evaluation config YAML.")
def evaluate_cmd(config: str) -> None:
    """Evaluates speaker verification accuracy, EER, and minDCF from config."""
    from swara.evaluation.harness import EvaluationHarness

    console.print(Panel.fit(f"[bold cyan]Swara Biometric Verification Evaluation[/bold cyan]\nConfig: {config}", border_style="cyan"))

    harness = EvaluationHarness()
    with console.status("Executing verification trials across evaluation dataset..."):
        metrics = harness.evaluate_from_config(config)

    table = Table(title="Evaluation Metrics Report", header_style="bold green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Total Evaluation Trials", str(metrics.get("total_trials", 0)))
    table.add_row("Target (Same Speaker) Trials", str(metrics.get("target_trials", 0)))
    table.add_row("Nontarget (Different Speaker) Trials", str(metrics.get("nontarget_trials", 0)))
    table.add_row("Equal Error Rate (EER)", f"{metrics.get('eer_percent', 0.0):.2f}%")
    table.add_row("Optimal Threshold", f"{metrics.get('optimal_threshold', 0.68):.4f}")
    table.add_row("Minimum DCF (minDCF)", f"{metrics.get('min_dcf', 0.0):.4f}")
    table.add_row("Mean Target Similarity", f"{metrics.get('mean_target_similarity', 0.0):.4f}")
    table.add_row("Mean Nontarget Similarity", f"{metrics.get('mean_nontarget_similarity', 0.0):.4f}")

    console.print(table)


@cli.command("serve")
@click.option("--host", default=settings.api_host, help="Bind host address.")
@click.option("--port", default=settings.api_port, type=int, help="Bind port number.")
@click.option("--workers", default=1, type=int, help="Number of worker processes.")
def serve_cmd(host: str, port: int, workers: int) -> None:
    """Starts the production HTTP REST API server."""
    import uvicorn
    console.print(f"[bold cyan]Starting Swara API server at http://{host}:{port}...[/bold cyan]")
    uvicorn.run("swara.api.app:app", host=host, port=port, workers=workers, log_level="info")


if __name__ == "__main__":
    cli()
