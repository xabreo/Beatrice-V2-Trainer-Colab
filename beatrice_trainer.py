"""
Beatrice V2 Simple Trainer — Python Backend

All wrapper logic for the five-tab Colab workflow.
The official Beatrice training engine is not modified.

Usage from notebook Cell 2:
    from beatrice_trainer import tab1_environment, tab2_project, ...
"""

import sys
import os
import json
import time
import shutil
import zipfile
import subprocess
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime


# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_TORCH = "2.8.0"
REQUIRED_TORCHAUDIO = "2.8.0"
REQUIRED_TORCHVISION = "0.23.0"
REQUIRED_PYWORLD = "0.3.4"
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu126"

BEATRICE_DRIVE = Path("/content/drive/MyDrive/Beatrice")
LOCAL_ROOT = Path("/content/beatrice_simple_trainer")
LOCAL_DATASET = LOCAL_ROOT / "dataset"
LOCAL_OUTPUT = LOCAL_ROOT / "training_output"
LOCAL_PROJECT = LOCAL_ROOT / "project"
LOCAL_LOGS = LOCAL_ROOT / "logs"
LOCAL_RUNTIME = LOCAL_ROOT / "runtime"
STATE_FILE = LOCAL_PROJECT / "state.json"
CONFIG_PATH = LOCAL_PROJECT / "trainer_config.json"
REPO = Path("/content/beatrice-trainer")

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}

WATCHDOG_INTERVAL = 30
NVIDIA_SMI_INTERVAL = 60


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _line():
    print("=" * 70)


def _section(title):
    print()
    _line()
    print(title)
    _line()


def _run_pip(args):
    cmd = [sys.executable, "-m", "pip"] + args
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError("pip command failed:\n" + " ".join(cmd))


def _audio_count(directory):
    return sum(
        1 for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def _safe_mtime(path):
    try:
        return path.stat().st_mtime
    except Exception:
        return 0


def _find_zip_candidates(folder):
    return sorted(
        [p for p in folder.rglob("*.zip") if p.is_file()],
        key=_safe_mtime,
        reverse=True
    )


def _find_latest_checkpoint(folder):
    candidates = [
        p for p in folder.rglob("checkpoint_latest.pt.gz")
        if p.is_file()
    ]
    return max(candidates, key=_safe_mtime) if candidates else None


def _now_text():
    return datetime.now().strftime("%H:%M:%S")


def _size_mb(path):
    try:
        return path.stat().st_size / 1024 / 1024
    except Exception:
        return 0.0


def _atomic_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_name(dst.name + ".copying")
    try:
        shutil.copy2(src, temp)
        temp.replace(dst)
        return True
    except Exception as e:
        print(f"[{_now_text()}] Backup failed: {src.name}: {e}")
        try:
            if temp.exists():
                temp.unlink()
        except Exception:
            pass
        return False


def _bridge_json(message, **extra):
    result = {"message": message}
    result.update(extra)
    return result


def _bridge_state():
    if not STATE_FILE.is_file():
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# TAB 1 — ENVIRONMENT
# ============================================================

def tab1_environment():
    """Environment installation + verification."""

    for folder in [LOCAL_ROOT, LOCAL_DATASET, LOCAL_OUTPUT, LOCAL_PROJECT, LOCAL_LOGS, LOCAL_RUNTIME]:
        folder.mkdir(parents=True, exist_ok=True)

    _section("1. PYTHON ENVIRONMENT")
    print("Python :", sys.version.split()[0])
    print("Executable:", sys.executable)

    _section("2. CHECKING PYTORCH ENVIRONMENT")
    torch_ok = False
    try:
        import torch
        import torchaudio
        import torchvision
        print("Current PyTorch    :", torch.__version__)
        print("Current TorchAudio :", torchaudio.__version__)
        print("Current TorchVision:", torchvision.__version__)
        current_torch = torch.__version__.split("+")[0]
        current_audio = torchaudio.__version__.split("+")[0]
        current_vision = torchvision.__version__.split("+")[0]
        if (current_torch == REQUIRED_TORCH
                and current_audio == REQUIRED_TORCHAUDIO
                and current_vision == REQUIRED_TORCHVISION):
            torch_ok = True
            print("\nKnown-good PyTorch environment already installed")
    except Exception as e:
        print("PyTorch environment is missing or incompatible:", e)

    if not torch_ok:
        _section("3. INSTALLING KNOWN-GOOD PYTORCH ENVIRONMENT")
        print("Removing current PyTorch stack...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                         "torch", "torchaudio", "torchvision"], text=True)
        print("\nInstalling:")
        print("  PyTorch    :", REQUIRED_TORCH)
        print("  TorchAudio :", REQUIRED_TORCHAUDIO)
        print("  TorchVision:", REQUIRED_TORCHVISION)
        print("  CUDA build : 12.6\n")
        _run_pip(["install", "--no-cache-dir",
                   f"torch=={REQUIRED_TORCH}",
                   f"torchaudio=={REQUIRED_TORCHAUDIO}",
                   f"torchvision=={REQUIRED_TORCHVISION}",
                   "--index-url", PYTORCH_INDEX])
        print("\nPyTorch installation complete")

    _section("4. PYWORLD")
    pyworld_ok = False
    try:
        import pyworld
        current_pyworld = getattr(pyworld, "__version__", "unknown")
        print("Current PyWorld :", current_pyworld)
        if current_pyworld == REQUIRED_PYWORLD:
            pyworld_ok = True
            print("PyWorld already correct")
    except Exception:
        print("PyWorld not available")
    if not pyworld_ok:
        print("\nInstalling PyWorld", REQUIRED_PYWORLD)
        _run_pip(["install", "--no-cache-dir", f"pyworld=={REQUIRED_PYWORLD}"])
        print("PyWorld installation complete")

    if not torch_ok:
        print()
        _line()
        print("RUNTIME RESTART REQUIRED")
        _line()
        print("\nThe correct Beatrice environment has been installed.")
        print("Please restart the Colab runtime:")
        print("    Runtime -> Restart session")
        print("\nThen run this SAME cell again.")
        _line()
    else:
        print("\nCorrect PyTorch environment already active")
        print("Continuing to environment verification...")

    try:
        import torch
        import torchaudio
        import torchvision
        import pyworld
    except Exception as e:
        raise RuntimeError(
            "\nEnvironment imports failed after installation.\n"
            "If PyTorch was just installed, restart the Colab runtime and run again.\n\n"
            f"Error: {e}"
        )

    _section("5. BEATRICE ENVIRONMENT VERIFICATION")
    print(f"Python          : {sys.version.split()[0]}")
    print(f"PyTorch         : {torch.__version__}")
    print(f"TorchAudio      : {torchaudio.__version__}")
    print(f"TorchVision     : {torchvision.__version__}")
    print(f"PyWorld         : {pyworld.__version__}")
    print(f"CUDA built      : {torch.version.cuda}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    _section("6. GPU SYSTEM CHECK")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print("GPU count       :", gpu_count)
        for i in range(gpu_count):
            print(f"GPU {i}           : {torch.cuda.get_device_name(i)}")
            print(f"VRAM {i}          : {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA is NOT available")
        print("Beatrice training requires a CUDA GPU.")

    _section("7. CUDA TEST")
    cuda_test_passed = False
    if torch.cuda.is_available():
        try:
            print("Running CUDA tensor test...")
            x = torch.randn(1024, 1024, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            del x, y
            torch.cuda.empty_cache()
            cuda_test_passed = True
            print("CUDA tensor test : PASSED")
        except Exception as e:
            print("CUDA tensor test : FAILED")
            print("Error:", e)
    else:
        print("CUDA tensor test : CUDA NOT AVAILABLE")

    _section("8. NVIDIA-SMI")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi failed")
            print(result.stderr)
    else:
        print("nvidia-smi not found")

    _section("9. GOOGLE DRIVE")
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount("/content/drive")
        print("\nGoogle Drive mounted")
    except Exception as e:
        raise RuntimeError(f"Google Drive mounting failed:\n{e}")

    _section("10. BEATRICE DRIVE DIRECTORY")
    print("Expected folder:", BEATRICE_DRIVE)
    if not BEATRICE_DRIVE.exists():
        print("\nBeatrice folder does not exist. Creating it...")
        BEATRICE_DRIVE.mkdir(parents=True, exist_ok=True)
        print("Created:", BEATRICE_DRIVE)
    else:
        print("Beatrice folder found")

    print("\nBeatrice Drive contents:")
    items = sorted(BEATRICE_DRIVE.iterdir())
    if items:
        for item in items:
            if item.is_dir():
                print(f"  {item.name}/")
            else:
                print(f"  {item.name} ({item.stat().st_size / 1024**2:.1f} MB)")
    else:
        print("  (empty)")

    _section("BEATRICE V2 SIMPLE TRAINER — TAB 1 STATUS")
    print("Python          : OK")
    print("PyTorch         :", "OK" if torch.__version__.split("+")[0] == REQUIRED_TORCH else "FAIL")
    print("TorchAudio      :", "OK" if torchaudio.__version__.split("+")[0] == REQUIRED_TORCHAUDIO else "FAIL")
    print("TorchVision     :", "OK" if torchvision.__version__.split("+")[0] == REQUIRED_TORCHVISION else "FAIL")
    print("PyWorld         :", "OK" if pyworld.__version__ == REQUIRED_PYWORLD else "FAIL")
    print("CUDA available  :", "OK" if torch.cuda.is_available() else "FAIL")
    print("CUDA test       :", "OK" if cuda_test_passed else "FAIL")
    print("Google Drive    : OK")
    print("Beatrice folder : OK")
    if torch.cuda.is_available():
        print("\nGPU             :", torch.cuda.get_device_name(0))
        print("VRAM            :", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    _line()
    print("TAB 1 — ENVIRONMENT READY")
    _line()
    print("\nNext step: Tab 2 — Project & Dataset Setup")


# ============================================================
# TAB 2 — PROJECT & DATASET DISCOVERY
# ============================================================

def tab2_project():
    """Drive + ZIP + dataset + fresh/resume."""

    for folder in [LOCAL_ROOT, LOCAL_DATASET, LOCAL_OUTPUT, LOCAL_PROJECT]:
        folder.mkdir(parents=True, exist_ok=True)

    _line()
    print("BEATRICE V2 SIMPLE TRAINER — TAB 2")
    print("PROJECT / DATASET DISCOVERY")
    _line()

    print("\nUSER DRIVE STRUCTURE")
    print("-" * 70)
    print("Required:")
    print("  My Drive/")
    print("    Beatrice/")
    print("      voices.zip                       (required)")
    print("      checkpoint_latest.pt.gz          (optional)")
    print("\nA project subfolder may also be used:")
    print("  Beatrice/")
    print("    MyVoice/")
    print("      voices.zip")
    print("      checkpoint_latest.pt.gz          (optional)")
    print()

    if not BEATRICE_DRIVE.is_dir():
        raise RuntimeError(
            "Beatrice folder not found.\n\nPlease create: My Drive -> Beatrice"
        )
    print("Beatrice Drive folder found")

    workspaces = []
    root_zips = _find_zip_candidates(BEATRICE_DRIVE)
    if root_zips:
        workspaces.append(BEATRICE_DRIVE)
    for child in sorted(BEATRICE_DRIVE.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir() and _find_zip_candidates(child):
            workspaces.append(child)

    if not workspaces:
        raise RuntimeError(
            "No voice dataset ZIP found inside Beatrice.\n\n"
            "Upload voices.zip into Beatrice or into a project subfolder."
        )

    def workspace_score(folder):
        zips = _find_zip_candidates(folder)
        preferred = [z for z in zips if z.name.lower() == "voices.zip"]
        chosen = preferred[0] if preferred else zips[0]
        return (1 if preferred else 0, _safe_mtime(chosen))

    selected_workspace = max(workspaces, key=workspace_score)
    zips = _find_zip_candidates(selected_workspace)
    preferred = [z for z in zips if z.name.lower() == "voices.zip"]
    DATASET_ZIP = preferred[0] if preferred else zips[0]
    RESUME_CHECKPOINT = _find_latest_checkpoint(selected_workspace)
    PROJECT_NAME = selected_workspace.name
    PROJECT_DRIVE = selected_workspace

    print("\n" + "=" * 70)
    print("1. WORKSPACE DISCOVERY")
    print("=" * 70)
    print("Selected workspace :", PROJECT_NAME)
    print("Workspace path     :", PROJECT_DRIVE)
    print("Dataset ZIP        :", DATASET_ZIP.name)
    if len(workspaces) > 1:
        print("\nOther dataset workspaces detected:")
        for ws in sorted(workspaces, key=lambda p: p.name.lower()):
            print("  *", ws.name)
    print("Workspace selected")

    print("\n" + "=" * 70)
    print("2. DATASET ZIP")
    print("=" * 70)
    print("ZIP:", DATASET_ZIP)
    print("Size:", f"{DATASET_ZIP.stat().st_size / 1024 / 1024:.1f} MB")
    print("Dataset ZIP found")

    if LOCAL_DATASET.exists():
        shutil.rmtree(LOCAL_DATASET)
    LOCAL_DATASET.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("3. EXTRACTING DATASET")
    print("=" * 70)
    try:
        with zipfile.ZipFile(DATASET_ZIP, "r") as z:
            bad = z.testzip()
            if bad:
                raise RuntimeError(f"Corrupt ZIP member: {bad}")
            z.extractall(LOCAL_DATASET)
    except zipfile.BadZipFile:
        raise RuntimeError(f"Invalid ZIP file: {DATASET_ZIP}")
    print("Dataset extracted")
    print("Local dataset:", LOCAL_DATASET)

    all_audio = sorted(
        [p for p in LOCAL_DATASET.rglob("*")
         if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    )
    if not all_audio:
        raise RuntimeError("No supported audio files found inside the dataset ZIP.")

    candidate_roots = []
    for directory in [LOCAL_DATASET] + [p for p in LOCAL_DATASET.rglob("*") if p.is_dir()]:
        count = _audio_count(directory)
        if count:
            candidate_roots.append((directory, count))
    DATASET_ROOT = min(
        candidate_roots,
        key=lambda item: (len(item[0].relative_to(LOCAL_DATASET).parts), -item[1])
    )[0]

    print("\n" + "=" * 70)
    print("4. DATASET STRUCTURE")
    print("=" * 70)
    print("Dataset root:", DATASET_ROOT)

    speaker_dirs = []
    for child in sorted(DATASET_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir():
            count = _audio_count(child)
            if count:
                speaker_dirs.append((child, count))

    direct_audio = [
        p for p in DATASET_ROOT.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]

    if direct_audio:
        speakers = [{"name": DATASET_ROOT.name, "path": str(DATASET_ROOT), "audio_count": len(all_audio)}]
        dataset_type = "single-speaker"
    elif speaker_dirs:
        speakers = [{"name": d.name, "path": str(d), "audio_count": count} for d, count in speaker_dirs]
        dataset_type = "multi-speaker"
    else:
        raise RuntimeError("Audio was found, but no usable speaker structure was detected.")

    speaker_count = len(speakers)
    total_audio = sum(s["audio_count"] for s in speakers)
    for speaker in speakers:
        print(f"{speaker['name']:<28}: {speaker['audio_count']:>6} audio files")
    print("\n" + "-" * 70)
    print("Dataset type :", dataset_type)
    print("Speakers     :", speaker_count)
    print("Total audio  :", total_audio)
    print("-" * 70)
    print("Dataset looks usable")

    print("\n" + "=" * 70)
    print("5. CHECKPOINT DISCOVERY")
    print("=" * 70)
    if RESUME_CHECKPOINT:
        print("Latest checkpoint:")
        print(" ", RESUME_CHECKPOINT)
        print("Size:", f"{RESUME_CHECKPOINT.stat().st_size / 1024 / 1024:.1f} MB")
        print("Modified:", datetime.fromtimestamp(RESUME_CHECKPOINT.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"))
        print("RESUME is available")
    else:
        print("No checkpoint_latest.pt.gz found in the selected workspace.")
        print("Fresh training is available.")

    TRAINING_MODE = "resume" if RESUME_CHECKPOINT else "fresh"

    state = {
        "version": "1.0",
        "project_name": PROJECT_NAME,
        "project_drive": str(PROJECT_DRIVE),
        "drive_root": str(BEATRICE_DRIVE),
        "dataset_zip": str(DATASET_ZIP),
        "dataset_zip_name": DATASET_ZIP.name,
        "dataset_root": str(DATASET_ROOT),
        "local_root": str(LOCAL_ROOT),
        "local_dataset": str(LOCAL_DATASET),
        "local_output": str(LOCAL_OUTPUT),
        "dataset_type": dataset_type,
        "speaker_count": speaker_count,
        "total_audio": total_audio,
        "speakers": speakers,
        "training_mode": TRAINING_MODE,
        "resume_checkpoint": str(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else None,
        "resume_available": bool(RESUME_CHECKPOINT),
        "updated_at": datetime.now().isoformat(),
        "created_by": "Beatrice V2 Simple Trainer"
    }

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("BEATRICE V2 SIMPLE TRAINER — TAB 2 STATUS")
    print("=" * 70)
    print("Workspace           :", PROJECT_NAME)
    print("Dataset ZIP         :", DATASET_ZIP.name)
    print("Dataset root        :", DATASET_ROOT)
    print("Dataset type        :", dataset_type)
    print("Speakers            :", speaker_count)
    print("Total audio         :", total_audio)
    print("Resume checkpoint   :", "AVAILABLE" if RESUME_CHECKPOINT else "NOT FOUND")
    print("Local workspace     :", LOCAL_ROOT)
    print("State file          :", STATE_FILE)
    print("\n" + "=" * 70)
    print("TAB 2 — PROJECT READY")
    print("=" * 70)
    print("\nTraining has not started.")
    print("The official Beatrice training engine is untouched.")
    print("Next step: TAB 3 — TRAINING CONFIGURATION")


# ============================================================
# TAB 3 — CONFIGURATION INITIALIZATION
# ============================================================

def tab3_initialize():
    """Create configuration/state."""

    _section("BEATRICE V2 SIMPLE TRAINER — TAB 3")
    print("TRAINING CONFIGURATION INITIALIZATION")

    _project_state = {}
    if STATE_FILE.is_file():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            _project_state = json.load(f)

    PROJECT_NAME = _project_state.get("project_name", "beatrice_project")
    PROJECT_DRIVE = Path(_project_state.get("project_drive", str(BEATRICE_DRIVE)))

    for folder in [LOCAL_ROOT, LOCAL_PROJECT, LOCAL_DATASET, LOCAL_OUTPUT, LOCAL_LOGS, LOCAL_RUNTIME]:
        folder.mkdir(parents=True, exist_ok=True)

    print("\nPROJECT")
    print("-" * 70)
    print(f"Beatrice Drive : {BEATRICE_DRIVE}")
    print(f"Project        : {PROJECT_NAME}")
    print(f"Project path   : {PROJECT_DRIVE}")
    print(f"Local workspace : {LOCAL_ROOT}")

    _section("1. PROJECT VERIFICATION")
    if not BEATRICE_DRIVE.is_dir():
        raise RuntimeError(f"Beatrice Drive folder not found: {BEATRICE_DRIVE}")
    print("Beatrice Drive folder found")
    if not PROJECT_DRIVE.is_dir():
        raise RuntimeError(f"Project folder not found: {PROJECT_DRIVE}")
    print("Project folder found")

    _section("2. HARDWARE DETECTION")
    try:
        CPU_CORES = os.cpu_count() or 1
    except Exception:
        CPU_CORES = 1
    print(f"CPU cores detected : {CPU_CORES}")

    GPU_AVAILABLE = False
    GPU_NAME = "CPU"
    GPU_VRAM_GB = 0.0
    try:
        import torch
        GPU_AVAILABLE = torch.cuda.is_available()
        if GPU_AVAILABLE:
            GPU_NAME = torch.cuda.get_device_name(0)
            GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception as e:
        print(f"GPU detection error: {e}")
    print(f"GPU available      : {GPU_AVAILABLE}")
    print(f"GPU                : {GPU_NAME}")
    print(f"VRAM               : {GPU_VRAM_GB:.2f} GB")

    _section("3. AUTO HARDWARE RECOMMENDATIONS")
    if CPU_CORES <= 2:
        AUTO_WORKERS = 2
    elif CPU_CORES <= 4:
        AUTO_WORKERS = 4
    elif CPU_CORES <= 8:
        AUTO_WORKERS = 8
    else:
        AUTO_WORKERS = min(CPU_CORES, 16)

    if GPU_VRAM_GB >= 20:
        AUTO_BATCH_SIZE = 16
    elif GPU_VRAM_GB >= 12:
        AUTO_BATCH_SIZE = 12
    elif GPU_VRAM_GB >= 8:
        AUTO_BATCH_SIZE = 8
    elif GPU_VRAM_GB >= 6:
        AUTO_BATCH_SIZE = 4
    else:
        AUTO_BATCH_SIZE = 2

    print(f"AUTO batch size : {AUTO_BATCH_SIZE}")
    print(f"AUTO workers    : {AUTO_WORKERS}")

    DEFAULT_CONFIG = {
        "training_mode": "fresh",
        "n_steps": 10000,
        "save_interval": 500,
        "evaluation_interval": 1000,
        "warmup_steps": 5000,
        "batch_size": AUTO_BATCH_SIZE,
        "num_workers": AUTO_WORKERS,
        "use_amp": True,
        "learning_rate_g": 5e-5,
        "learning_rate_d": 5e-5,
        "learning_rate_decay": 0.999999,
        "adam_betas": [0.8, 0.99],
        "adam_eps": 1e-6,
        "grad_weight_loudness": 1.0,
        "grad_weight_mel": 50.0,
        "grad_weight_ap": 100.0,
        "grad_weight_adv": 150.0,
        "grad_weight_fm": 150.0,
        "grad_balancer_ema_decay": 0.995,
        "in_sample_rate": 16000,
        "out_sample_rate": 24000,
        "wav_length": 96000,
        "segment_length": 100,
        "phone_noise_ratio": 0.5,
        "vq_topk": 4,
        "training_time_vq": "none",
        "floor_noise_level": 1e-3,
        "augmentation_snr_candidates": [20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
        "augmentation_formant_shift_probability": 0.5,
        "augmentation_formant_shift_semitone_min": -3.0,
        "augmentation_formant_shift_semitone_max": 3.0,
        "augmentation_reverb_probability": 0.5,
        "augmentation_lpf_probability": 0.2,
        "augmentation_lpf_cutoff_freq_candidates": [2000.0, 3000.0, 4000.0, 6000.0],
        "pitch_bins": 448,
        "hidden_channels": 256,
        "record_metrics": False,
        "san": False,
        "compile_convnext": False,
        "compile_d4c": False,
        "compile_discriminator": False,
        "profile": False,
        "hardware": {
            "gpu_name": GPU_NAME,
            "gpu_vram_gb": round(GPU_VRAM_GB, 2),
            "cpu_cores": CPU_CORES,
            "auto_batch_size": AUTO_BATCH_SIZE,
            "auto_num_workers": AUTO_WORKERS,
        },
        "wrapper": {
            "name": "BEATRICE V2 SIMPLE TRAINER",
            "version": "1.0",
            "created": datetime.now().isoformat(),
        }
    }

    _section("5. CONFIGURATION INITIALIZATION")
    if CONFIG_PATH.is_file():
        print("Existing trainer_config.json found.")
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                CONFIG = json.load(f)
            print("Existing configuration loaded")
        except Exception as e:
            print(f"Existing configuration could not be loaded: {e}")
            print("Creating a new configuration...")
            CONFIG = DEFAULT_CONFIG.copy()
    else:
        print("No existing trainer_config.json found.")
        print("Creating default configuration...")
        CONFIG = DEFAULT_CONFIG

    CONFIG["hardware"] = {
        "gpu_name": GPU_NAME,
        "gpu_vram_gb": round(GPU_VRAM_GB, 2),
        "cpu_cores": CPU_CORES,
        "auto_batch_size": AUTO_BATCH_SIZE,
        "auto_num_workers": AUTO_WORKERS,
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4)
    print("trainer_config.json saved")

    state = {
        "project_name": PROJECT_NAME,
        "project_drive": str(PROJECT_DRIVE),
        "local_workspace": str(LOCAL_ROOT),
        "dataset_path": str(LOCAL_DATASET),
        "training_output": str(LOCAL_OUTPUT),
        "training_mode": CONFIG.get("training_mode", "fresh"),
        "config_path": str(CONFIG_PATH),
        "created_or_updated": datetime.now().isoformat(),
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)
    print("state.json saved")

    _section("6. NORMAL USER SETTINGS")
    print("Training mode       :", CONFIG.get("training_mode"))
    print("Training steps      :", CONFIG.get("n_steps"))
    print("Save interval       :", CONFIG.get("save_interval"))
    print("Evaluation interval :", CONFIG.get("evaluation_interval"))
    print("Warmup steps        :", CONFIG.get("warmup_steps"))
    print("Batch size          :", CONFIG.get("batch_size"))
    print("CPU workers         :", CONFIG.get("num_workers"))
    print("AMP                 :", CONFIG.get("use_amp"))

    _section("7. PRO SETTINGS")
    print("Learning rate G             :", CONFIG.get("learning_rate_g"))
    print("Learning rate D             :", CONFIG.get("learning_rate_d"))
    print("Learning rate decay         :", CONFIG.get("learning_rate_decay"))
    print("Adam betas                  :", CONFIG.get("adam_betas"))
    print("Adam epsilon                :", CONFIG.get("adam_eps"))
    print("Mel loss weight             :", CONFIG.get("grad_weight_mel"))
    print("AP loss weight              :", CONFIG.get("grad_weight_ap"))
    print("Adversarial loss weight     :", CONFIG.get("grad_weight_adv"))
    print("Feature matching weight     :", CONFIG.get("grad_weight_fm"))
    print("Input sample rate           :", CONFIG.get("in_sample_rate"))
    print("Output sample rate          :", CONFIG.get("out_sample_rate"))
    print("Segment length              :", CONFIG.get("segment_length"))
    print("VQ Top-K                    :", CONFIG.get("vq_topk"))
    print("Pitch bins                  :", CONFIG.get("pitch_bins"))
    print("Hidden channels             :", CONFIG.get("hidden_channels"))

    _section("8. AUGMENTATION")
    print("SNR candidates        :", CONFIG.get("augmentation_snr_candidates"))
    print("Formant shift probability :", CONFIG.get("augmentation_formant_shift_probability"))
    print("Formant shift range       :", CONFIG.get("augmentation_formant_shift_semitone_min"), "to", CONFIG.get("augmentation_formant_shift_semitone_max"))
    print("Reverb probability        :", CONFIG.get("augmentation_reverb_probability"))
    print("LPF probability           :", CONFIG.get("augmentation_lpf_probability"))

    _section("BEATRICE V2 SIMPLE TRAINER — TAB 3 STATUS")
    print("Project             :", PROJECT_NAME)
    print("Workspace           :", LOCAL_ROOT)
    print("Configuration       :", CONFIG_PATH.name)
    print("State               :", STATE_FILE.name)
    print("GPU                 :", GPU_NAME)
    print("VRAM                :", f"{GPU_VRAM_GB:.2f} GB")
    print("CPU cores           :", CPU_CORES)
    print("AUTO batch size     :", AUTO_BATCH_SIZE)
    print("AUTO workers        :", AUTO_WORKERS)
    print("Training mode       :", CONFIG.get("training_mode"))
    print("Training steps      :", CONFIG.get("n_steps"))
    print("\n" + "=" * 70)
    print("TAB 3 — CONFIGURATION READY")
    print("=" * 70)
    print("\nNo training has been started.")
    print("No Beatrice training logic has been modified.")
    print("\nNext step: TAB 4 — PRO SETTINGS / FINAL CONFIGURATION")


# ============================================================
# TAB 4 — SETTINGS (NORMAL + PRO)
# ============================================================

def tab4_settings():
    """Normal + pro settings."""

    _section("BEATRICE V2 SIMPLE TRAINER — TAB 4")
    print("TRAINING SETTINGS")

    _section("1. CONFIGURATION CHECK")
    print("Configuration :", CONFIG_PATH)
    if not CONFIG_PATH.is_file():
        raise RuntimeError("trainer_config.json not found. Please run TAB 3 first.")
    print("trainer_config.json found")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
    print("Configuration loaded")

    HARDWARE = CONFIG.get("hardware", {})
    GPU_NAME = HARDWARE.get("gpu_name", "Unknown")
    GPU_VRAM = HARDWARE.get("gpu_vram_gb", 0)
    CPU_CORES = HARDWARE.get("cpu_cores", 1)
    AUTO_BATCH = HARDWARE.get("auto_batch_size", CONFIG.get("batch_size", 12))
    AUTO_WORKERS = HARDWARE.get("auto_num_workers", CONFIG.get("num_workers", 2))

    _section("2. NORMAL USER SETTINGS")
    TRAINING_MODE = CONFIG.get("training_mode", "fresh")
    if TRAINING_MODE not in {"fresh", "resume"}:
        TRAINING_MODE = "fresh"
    N_STEPS = int(CONFIG.get("n_steps", 10000))
    SAVE_INTERVAL = int(CONFIG.get("save_interval", 500))
    EVALUATION_INTERVAL = int(CONFIG.get("evaluation_interval", 1000))
    WARMUP_STEPS = int(CONFIG.get("warmup_steps", 5000))
    BATCH_SIZE = int(CONFIG.get("batch_size", AUTO_BATCH))
    NUM_WORKERS = int(CONFIG.get("num_workers", AUTO_WORKERS))
    USE_AMP = bool(CONFIG.get("use_amp", True))

    print("Training mode       :", TRAINING_MODE.upper())
    print("Training steps      :", N_STEPS)
    print("Save interval       :", SAVE_INTERVAL)
    print("Evaluation interval :", EVALUATION_INTERVAL)
    print("Warmup steps        :", WARMUP_STEPS)
    print("Batch size          :", BATCH_SIZE)
    print("CPU workers         :", NUM_WORKERS)
    print("AMP                 :", USE_AMP)

    _section("3. HARDWARE")
    print("GPU          :", GPU_NAME)
    print("VRAM         :", f"{GPU_VRAM:.2f} GB")
    print("CPU cores    :", CPU_CORES)
    print("AUTO batch   :", AUTO_BATCH)
    print("AUTO workers :", AUTO_WORKERS)

    _section("4. PRO SETTINGS")
    LEARNING_RATE_G = float(CONFIG.get("learning_rate_g", 5e-5))
    LEARNING_RATE_D = float(CONFIG.get("learning_rate_d", 5e-5))
    LEARNING_RATE_DECAY = float(CONFIG.get("learning_rate_decay", 0.999999))
    ADAM_BETAS = CONFIG.get("adam_betas", [0.8, 0.99])
    ADAM_EPS = float(CONFIG.get("adam_eps", 1e-6))
    GRAD_WEIGHT_LOUDNESS = float(CONFIG.get("grad_weight_loudness", 1.0))
    GRAD_WEIGHT_MEL = float(CONFIG.get("grad_weight_mel", 50.0))
    GRAD_WEIGHT_AP = float(CONFIG.get("grad_weight_ap", 100.0))
    GRAD_WEIGHT_ADV = float(CONFIG.get("grad_weight_adv", 150.0))
    GRAD_WEIGHT_FM = float(CONFIG.get("grad_weight_fm", 150.0))
    GRAD_BALANCER_EMA_DECAY = float(CONFIG.get("grad_balancer_ema_decay", 0.995))
    IN_SAMPLE_RATE = int(CONFIG.get("in_sample_rate", 16000))
    OUT_SAMPLE_RATE = int(CONFIG.get("out_sample_rate", 24000))
    WAV_LENGTH = int(CONFIG.get("wav_length", 96000))
    SEGMENT_LENGTH = int(CONFIG.get("segment_length", 100))
    PHONE_NOISE_RATIO = float(CONFIG.get("phone_noise_ratio", 0.5))
    VQ_TOPK = int(CONFIG.get("vq_topk", 4))
    TRAINING_TIME_VQ = CONFIG.get("training_time_vq", "none")
    FLOOR_NOISE_LEVEL = float(CONFIG.get("floor_noise_level", 1e-3))
    AUG_SNR = CONFIG.get("augmentation_snr_candidates", [20.0, 25.0, 30.0, 35.0, 40.0, 45.0])
    AUG_FORMANT_PROB = float(CONFIG.get("augmentation_formant_shift_probability", 0.5))
    AUG_FORMANT_MIN = float(CONFIG.get("augmentation_formant_shift_semitone_min", -3.0))
    AUG_FORMANT_MAX = float(CONFIG.get("augmentation_formant_shift_semitone_max", 3.0))
    AUG_REVERB_PROB = float(CONFIG.get("augmentation_reverb_probability", 0.5))
    AUG_LPF_PROB = float(CONFIG.get("augmentation_lpf_probability", 0.2))
    AUG_LPF_CUTOFF = CONFIG.get("augmentation_lpf_cutoff_freq_candidates", [2000.0, 3000.0, 4000.0, 6000.0])
    PITCH_BINS = int(CONFIG.get("pitch_bins", 448))
    HIDDEN_CHANNELS = int(CONFIG.get("hidden_channels", 256))
    RECORD_METRICS = bool(CONFIG.get("record_metrics", False))
    SAN = bool(CONFIG.get("san", False))
    COMPILE_CONVNEXT = bool(CONFIG.get("compile_convnext", False))
    COMPILE_D4C = bool(CONFIG.get("compile_d4c", False))
    COMPILE_DISCRIMINATOR = bool(CONFIG.get("compile_discriminator", False))
    PROFILE = bool(CONFIG.get("profile", False))

    print("\nLEARNING")
    print("-" * 70)
    print("Learning rate G       :", LEARNING_RATE_G)
    print("Learning rate D       :", LEARNING_RATE_D)
    print("Learning rate decay   :", LEARNING_RATE_DECAY)
    print("Adam betas            :", ADAM_BETAS)
    print("Adam epsilon          :", ADAM_EPS)
    print("\nLOSS WEIGHTS")
    print("-" * 70)
    print("Loudness              :", GRAD_WEIGHT_LOUDNESS)
    print("Mel                   :", GRAD_WEIGHT_MEL)
    print("AP                    :", GRAD_WEIGHT_AP)
    print("Adversarial           :", GRAD_WEIGHT_ADV)
    print("Feature matching      :", GRAD_WEIGHT_FM)
    print("Balancer EMA decay    :", GRAD_BALANCER_EMA_DECAY)
    print("\nAUDIO")
    print("-" * 70)
    print("Input sample rate     :", IN_SAMPLE_RATE)
    print("Output sample rate    :", OUT_SAMPLE_RATE)
    print("WAV length            :", WAV_LENGTH)
    print("Segment length        :", SEGMENT_LENGTH)
    print("Phone noise ratio     :", PHONE_NOISE_RATIO)
    print("VQ Top-K              :", VQ_TOPK)
    print("Training time VQ      :", TRAINING_TIME_VQ)
    print("Floor noise level     :", FLOOR_NOISE_LEVEL)
    print("\nMODEL")
    print("-" * 70)
    print("Pitch bins            :", PITCH_BINS)
    print("Hidden channels       :", HIDDEN_CHANNELS)
    print("\nPERFORMANCE / DEBUG")
    print("-" * 70)
    print("Record metrics        :", RECORD_METRICS)
    print("SAN                   :", SAN)
    print("Compile ConvNeXt      :", COMPILE_CONVNEXT)
    print("Compile D4C           :", COMPILE_D4C)
    print("Compile discriminator :", COMPILE_DISCRIMINATOR)
    print("Profile               :", PROFILE)

    _section("5. SETTINGS VALIDATION")
    ERRORS = []
    WARNINGS = []
    if N_STEPS <= 0:
        ERRORS.append("Training steps must be greater than 0.")
    if SAVE_INTERVAL <= 0:
        ERRORS.append("Save interval must be greater than 0.")
    if SAVE_INTERVAL > N_STEPS:
        WARNINGS.append("Save interval is larger than total training steps.")
    if EVALUATION_INTERVAL <= 0:
        ERRORS.append("Evaluation interval must be greater than 0.")
    if WARMUP_STEPS < 0:
        ERRORS.append("Warmup steps cannot be negative.")
    if WARMUP_STEPS > N_STEPS:
        WARNINGS.append("Warmup steps exceed total training steps.")
    if BATCH_SIZE <= 0:
        ERRORS.append("Batch size must be greater than 0.")
    if NUM_WORKERS < 0:
        ERRORS.append("CPU workers cannot be negative.")
    if NUM_WORKERS > CPU_CORES:
        WARNINGS.append(f"CPU workers ({NUM_WORKERS}) exceed detected CPU cores ({CPU_CORES}).")
    if LEARNING_RATE_G <= 0:
        ERRORS.append("Generator learning rate must be greater than 0.")
    if LEARNING_RATE_D <= 0:
        ERRORS.append("Discriminator learning rate must be greater than 0.")
    if IN_SAMPLE_RATE <= 0:
        ERRORS.append("Input sample rate must be greater than 0.")
    if OUT_SAMPLE_RATE <= 0:
        ERRORS.append("Output sample rate must be greater than 0.")
    if AUG_FORMANT_MIN > AUG_FORMANT_MAX:
        ERRORS.append("Formant minimum cannot be greater than maximum.")
    for name, value in {"Formant shift": AUG_FORMANT_PROB, "Reverb": AUG_REVERB_PROB, "LPF": AUG_LPF_PROB}.items():
        if not 0 <= value <= 1:
            ERRORS.append(f"{name} probability must be between 0 and 1.")

    if ERRORS:
        print("\nCONFIGURATION ERRORS")
        for error in ERRORS:
            print("  ERROR:", error)
        raise RuntimeError("Configuration validation failed.")
    print("No blocking configuration errors")
    if WARNINGS:
        print("\nWARNINGS")
        for warning in WARNINGS:
            print("  WARNING:", warning)
    else:
        print("No configuration warnings")

    CONFIG["training_mode"] = TRAINING_MODE
    CONFIG["n_steps"] = N_STEPS
    CONFIG["save_interval"] = SAVE_INTERVAL
    CONFIG["evaluation_interval"] = EVALUATION_INTERVAL
    CONFIG["warmup_steps"] = WARMUP_STEPS
    CONFIG["batch_size"] = BATCH_SIZE
    CONFIG["num_workers"] = NUM_WORKERS
    CONFIG["use_amp"] = USE_AMP
    CONFIG["learning_rate_g"] = LEARNING_RATE_G
    CONFIG["learning_rate_d"] = LEARNING_RATE_D
    CONFIG["learning_rate_decay"] = LEARNING_RATE_DECAY
    CONFIG["adam_betas"] = ADAM_BETAS
    CONFIG["adam_eps"] = ADAM_EPS
    CONFIG["grad_weight_loudness"] = GRAD_WEIGHT_LOUDNESS
    CONFIG["grad_weight_mel"] = GRAD_WEIGHT_MEL
    CONFIG["grad_weight_ap"] = GRAD_WEIGHT_AP
    CONFIG["grad_weight_adv"] = GRAD_WEIGHT_ADV
    CONFIG["grad_weight_fm"] = GRAD_WEIGHT_FM
    CONFIG["grad_balancer_ema_decay"] = GRAD_BALANCER_EMA_DECAY
    CONFIG["in_sample_rate"] = IN_SAMPLE_RATE
    CONFIG["out_sample_rate"] = OUT_SAMPLE_RATE
    CONFIG["wav_length"] = WAV_LENGTH
    CONFIG["segment_length"] = SEGMENT_LENGTH
    CONFIG["phone_noise_ratio"] = PHONE_NOISE_RATIO
    CONFIG["vq_topk"] = VQ_TOPK
    CONFIG["training_time_vq"] = TRAINING_TIME_VQ
    CONFIG["floor_noise_level"] = FLOOR_NOISE_LEVEL
    CONFIG["augmentation_snr_candidates"] = AUG_SNR
    CONFIG["augmentation_formant_shift_probability"] = AUG_FORMANT_PROB
    CONFIG["augmentation_formant_shift_semitone_min"] = AUG_FORMANT_MIN
    CONFIG["augmentation_formant_shift_semitone_max"] = AUG_FORMANT_MAX
    CONFIG["augmentation_reverb_probability"] = AUG_REVERB_PROB
    CONFIG["augmentation_lpf_probability"] = AUG_LPF_PROB
    CONFIG["augmentation_lpf_cutoff_freq_candidates"] = AUG_LPF_CUTOFF
    CONFIG["pitch_bins"] = PITCH_BINS
    CONFIG["hidden_channels"] = HIDDEN_CHANNELS
    CONFIG["record_metrics"] = RECORD_METRICS
    CONFIG["san"] = SAN
    CONFIG["compile_convnext"] = COMPILE_CONVNEXT
    CONFIG["compile_d4c"] = COMPILE_D4C
    CONFIG["compile_discriminator"] = COMPILE_DISCRIMINATOR
    CONFIG["profile"] = PROFILE
    CONFIG["hardware"] = {
        "gpu_name": GPU_NAME, "gpu_vram_gb": GPU_VRAM,
        "cpu_cores": CPU_CORES, "auto_batch_size": AUTO_BATCH, "auto_num_workers": AUTO_WORKERS,
    }
    CONFIG["wrapper"] = {
        "name": "BEATRICE V2 SIMPLE TRAINER", "version": "1.0",
        "last_modified": datetime.now().isoformat(),
    }

    _section("6. SAVING CONFIGURATION")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4)
    print("trainer_config.json updated")

    state = {}
    if STATE_FILE.is_file():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            state = {}
    state.update({
        "training_mode": TRAINING_MODE, "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS,
        "config_path": str(CONFIG_PATH),
        "last_settings_update": datetime.now().isoformat(),
    })
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)
    print("state.json updated")

    _section("BEATRICE V2 SIMPLE TRAINER — TAB 4 STATUS")
    print("Project          :", state.get("project_name", "example_project"))
    print("Training mode    :", TRAINING_MODE.upper())
    print("Training steps   :", N_STEPS)
    print("Save interval    :", SAVE_INTERVAL)
    print("Evaluation       :", EVALUATION_INTERVAL)
    print("Warmup           :", WARMUP_STEPS)
    print("Batch size       :", BATCH_SIZE)
    print("CPU workers      :", NUM_WORKERS)
    print("AMP              :", USE_AMP)
    print("Configuration    :", CONFIG_PATH)
    print("\n" + "=" * 70)
    print("TAB 4 — SETTINGS READY")
    print("=" * 70)
    print("\nFinal configuration saved.")
    print("No training has been started.")
    print("No Beatrice training logic has been modified.")
    print("\nNext step: TAB 5 — START TRAINING + LIVE WATCHDOG")


# ============================================================
# TAB 5 — TRAINING + LIVE WATCHDOG
# ============================================================

def tab5_training():
    """Beatrice + watchdog + backup."""

    if not STATE_FILE.is_file():
        raise RuntimeError("state.json not found. Run TAB 2 first.")
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    PROJECT_NAME = state["project_name"]
    PROJECT_DRIVE = Path(state["project_drive"])
    DATASET_ROOT = Path(state["dataset_root"])
    TRAINING_MODE = state.get("training_mode", "fresh")

    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    PROJECT_DRIVE.mkdir(parents=True, exist_ok=True)

    _section("BEATRICE V2 SIMPLE TRAINER — TAB 5")
    print("START TRAINING + LIVE WATCHDOG")
    print("Workspace     :", PROJECT_NAME)
    print("Drive folder  :", PROJECT_DRIVE)
    print("Beatrice repo :", REPO)
    print("Local output  :", LOCAL_OUTPUT)

    _section("1. FINAL PRE-FLIGHT")
    if not BEATRICE_DRIVE.is_dir():
        raise RuntimeError(f"Beatrice folder not found: {BEATRICE_DRIVE}")
    print("Beatrice Drive folder found")
    if not PROJECT_DRIVE.is_dir():
        raise RuntimeError(f"Selected workspace not found: {PROJECT_DRIVE}")
    print("Workspace found")
    if not CONFIG_PATH.is_file():
        raise RuntimeError(f"trainer_config.json not found: {CONFIG_PATH}")
    print("trainer_config.json found")
    if not DATASET_ROOT.is_dir():
        raise RuntimeError(f"Dataset root not found: {DATASET_ROOT}")
    print("Dataset root found")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print("Configuration loaded")
    print("Training mode :", TRAINING_MODE)
    print("Training steps:", cfg.get("n_steps"))
    print("Batch size    :", cfg.get("batch_size"))
    print("CPU workers   :", cfg.get("num_workers"))
    print("AMP           :", cfg.get("use_amp"))

    _section("2. DATASET PRE-FLIGHT")
    audio_files = [p for p in DATASET_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    if not audio_files:
        raise RuntimeError("No audio files found in dataset.")
    speaker_dirs = []
    for child in sorted(DATASET_ROOT.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir():
            n = sum(1 for p in child.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
            if n:
                speaker_dirs.append((child.name, n))
    if speaker_dirs:
        for name, n in speaker_dirs:
            print(f"{name:<28}: {n:>6} audio files")
    else:
        print(f"{DATASET_ROOT.name:<28}: {len(audio_files):>6} audio files")
    print("Total audio   :", len(audio_files))
    print("Dataset ready")

    _section("3. BEATRICE TRAINER CHECK")
    BEATRICE_REPO_URL = "https://huggingface.co/fierce-cats/beatrice-trainer"
    if not REPO.is_dir():
        print("Official Beatrice trainer not found.")
        print("Preparing official Beatrice trainer...")
        print("Repository:", BEATRICE_REPO_URL)
        lfs = shutil.which("git-lfs")
        if lfs is None:
            print("Installing Git LFS...")
            install = subprocess.run(["apt-get", "update", "-qq"], text=True)
            if install.returncode != 0:
                raise RuntimeError("Could not update apt for Git LFS.")
            install = subprocess.run(["apt-get", "install", "-y", "-qq", "git-lfs"], text=True)
            if install.returncode != 0:
                raise RuntimeError("Could not install Git LFS.")
        subprocess.run(["git", "lfs", "install"], check=True)
        print("Git LFS ready")
        clone_result = subprocess.run(["git", "clone", BEATRICE_REPO_URL, str(REPO)], text=True)
        if clone_result.returncode != 0:
            raise RuntimeError("Failed to obtain official Beatrice trainer.")
        print("Official Beatrice trainer downloaded")
    else:
        print("Official Beatrice trainer already available")
    trainer_module = REPO / "beatrice_trainer"
    if not trainer_module.is_dir():
        raise RuntimeError(f"Invalid Beatrice repository: {REPO}")
    print("Official Beatrice trainer verified")

    _section("4. BEATRICE CONFIGURATION")
    BEATRICE_CONFIG = REPO / "assets" / "simple_trainer_config.json"
    beatrice_cfg = dict(cfg)
    beatrice_cfg["data_dir"] = str(DATASET_ROOT)
    beatrice_cfg["out_dir"] = str(LOCAL_OUTPUT)
    for key in {"training_mode", "project_name", "project_drive", "local_root",
                "local_dataset", "local_output", "watchdog_interval",
                "checkpoint_history_to_keep", "keep_paraphernalia", "keep_history", "drive_sync"}:
        beatrice_cfg.pop(key, None)
    BEATRICE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(BEATRICE_CONFIG, "w", encoding="utf-8") as f:
        json.dump(beatrice_cfg, f, indent=4)
    print("Config:", BEATRICE_CONFIG)
    print("Data  :", DATASET_ROOT)
    print("Output:", LOCAL_OUTPUT)
    print("Configuration bridge ready")

    _section("5. TRAINING MODE")
    local_latest = LOCAL_OUTPUT / "checkpoint_latest.pt.gz"
    drive_latest = PROJECT_DRIVE / "checkpoint_latest.pt.gz"
    for item in LOCAL_OUTPUT.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    if TRAINING_MODE == "resume":
        if not drive_latest.is_file():
            raise RuntimeError(
                "RESUME selected, but checkpoint_latest.pt.gz was not found in the selected Drive workspace."
            )
        shutil.copy2(drive_latest, local_latest)
        print("Mode : RESUME")
        print("Resume checkpoint restored:", local_latest)
    else:
        print("Mode : FRESH")
        print("Existing Drive checkpoints will not be used as the training base.")

    _section("6. WATCHDOG INITIALIZATION")
    processed_checkpoint_mtimes = {}
    processed_paraphernalia = set()

    def numbered_checkpoints():
        return sorted(
            [p for p in LOCAL_OUTPUT.glob("checkpoint_*.pt.gz")
             if p.is_file() and p.name != "checkpoint_latest.pt.gz"],
            key=lambda p: p.stat().st_mtime
        )

    def paraphernalia_folders():
        return sorted(
            [p for p in LOCAL_OUTPUT.glob("paraphernalia_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime
        )

    def backup_checkpoints():
        cps = numbered_checkpoints()
        if not cps:
            print(f"[{_now_text()}] Watchdog: No numbered checkpoints yet")
            return
        print(f"\n{'=' * 70}\n[{_now_text()}] CHECKPOINT WATCHDOG\n{'=' * 70}")
        print("Checkpoints found:", len(cps))
        for cp in cps:
            mtime = cp.stat().st_mtime
            if processed_checkpoint_mtimes.get(cp.name) == mtime:
                continue
            dst = PROJECT_DRIVE / cp.name
            if dst.is_file() and dst.stat().st_size == cp.stat().st_size:
                processed_checkpoint_mtimes[cp.name] = mtime
                print(f"[{_now_text()}] Already on Drive: {cp.name}")
            elif _atomic_copy(cp, dst):
                processed_checkpoint_mtimes[cp.name] = mtime
                print(f"[{_now_text()}] BACKED UP: {cp.name} ({_size_mb(cp):.1f} MB)")
        newest = cps[-1]
        latest_dst = PROJECT_DRIVE / "checkpoint_latest.pt.gz"
        update_latest = True
        if latest_dst.is_file():
            try:
                update_latest = (latest_dst.stat().st_size != newest.stat().st_size
                                or latest_dst.stat().st_mtime < newest.stat().st_mtime)
            except Exception:
                update_latest = True
        if update_latest and _atomic_copy(newest, latest_dst):
            print(f"[{_now_text()}] UPDATED LATEST: {newest.name} -> checkpoint_latest.pt.gz")
        print("Historical numbered checkpoints: KEEP ALL")

    def backup_paraphernalia():
        folders = paraphernalia_folders()
        if not folders:
            print(f"[{_now_text()}] No paraphernalia folders yet")
            return
        print(f"\n{'=' * 70}\n[{_now_text()}] PARAPHERNALIA WATCHDOG\n{'=' * 70}")
        for folder in folders:
            if folder.name in processed_paraphernalia:
                continue
            dst = PROJECT_DRIVE / folder.name
            if dst.is_dir():
                processed_paraphernalia.add(folder.name)
                print(f"[{_now_text()}] Already on Drive: {folder.name}")
                continue
            try:
                shutil.copytree(folder, dst)
                processed_paraphernalia.add(folder.name)
                print(f"[{_now_text()}] BACKED UP PARAPHERNALIA: {folder.name}")
            except Exception as e:
                print(f"[{_now_text()}] Paraphernalia backup failed: {folder.name}: {e}")

    def run_nvidia_smi():
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        print(f"\n{'=' * 70}\n[{_now_text()}] NVIDIA-SMI\n{'=' * 70}")
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(result.stderr)

    stop_watchdogs = threading.Event()

    def watchdog_loop():
        next_gpu = time.time()
        while not stop_watchdogs.is_set():
            try:
                backup_checkpoints()
                backup_paraphernalia()
                if time.time() >= next_gpu:
                    run_nvidia_smi()
                    next_gpu = time.time() + NVIDIA_SMI_INTERVAL
            except Exception as e:
                print(f"[{_now_text()}] Watchdog error: {e}")
            stop_watchdogs.wait(WATCHDOG_INTERVAL)

    watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)

    _section("7. STARTING OFFICIAL BEATRICE")
    cmd = [sys.executable, "-m", "beatrice_trainer", "-c", str(BEATRICE_CONFIG)]
    if TRAINING_MODE == "resume":
        cmd.append("-r")
    print("Launching:")
    print(" ".join(cmd))
    print("\n30-second checkpoint watchdog : ACTIVE")
    print("60-second nvidia-smi monitor   : ACTIVE")
    print("Historical checkpoints         : KEEP ALL")
    print("paraphernalia_*                : KEEP ALL")
    print("Drive checkpoint_latest        : ONE rolling copy")

    watchdog_thread.start()

    process = subprocess.Popen(
        cmd, cwd=REPO, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    return_code = None
    try:
        while True:
            line = process.stdout.readline()
            if line:
                print(line, end="", flush=True)
            if process.poll() is not None:
                return_code = process.returncode
                break
            time.sleep(0.1)
        print(f"\n{'=' * 70}\n[{_now_text()}] FINAL BACKUP\n{'=' * 70}")
        backup_checkpoints()
        backup_paraphernalia()
    finally:
        stop_watchdogs.set()
        watchdog_thread.join(timeout=5)

    print("\n" + "=" * 70)
    if return_code == 0:
        print("BEATRICE V2 SIMPLE TRAINER")
        print("TRAINING COMPLETED SUCCESSFULLY")
    else:
        print("BEATRICE V2 SIMPLE TRAINER")
        print("TRAINING PROCESS EXITED WITH ERROR")
    print("=" * 70)
    print("Workspace        :", PROJECT_NAME)
    print("Training mode    :", TRAINING_MODE.upper())
    print("Local output     :", LOCAL_OUTPUT)
    print("Drive workspace  :", PROJECT_DRIVE)
    print("Paraphernalia    : KEEP ALL")
    print("Numbered checkpoints: KEEP ALL")
    print("Latest checkpoint:", PROJECT_DRIVE / "checkpoint_latest.pt.gz")
    print("\nThe official Beatrice training engine was not modified.")

    if return_code != 0:
        raise RuntimeError(f"Beatrice training exited with code {return_code}")
