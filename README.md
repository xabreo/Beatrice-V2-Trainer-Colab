# Beatrice-V2-Trainer-Colab
# Beatrice V2 Simple Trainer

### A simple Google Colab wrapper for training Beatrice V2 voice models

Beatrice V2 Simple Trainer is a **community-made convenience wrapper** designed to make the existing Beatrice V2 training workflow easier for people who are not familiar with Python, command-line tools, configuration files, or manual checkpoint management.

**It does not replace Beatrice V2.**

**It does not modify or reimplement the Beatrice V2 training engine.**

**It does not claim ownership of Beatrice V2 or its underlying training technology.**

The actual voice-model training is performed by the original **Beatrice V2 Trainer**.

Our goal is simple:

> **Make the existing Beatrice V2 training workflow easier for everyone to use.**

---

## What this project does

The original Beatrice training workflow is powerful, but using it directly can require familiarity with:

* Python environments
* PyTorch and CUDA versions
* Python packages
* Google Colab
* Google Drive
* JSON configuration files
* Dataset directory structures
* Training checkpoints
* Resume commands
* Training output management

This project puts a simple layer around those steps.

Instead of manually managing the entire workflow, the user can follow a guided five-tab process:

### 1. Environment

Automatically prepares and verifies the required Colab environment.

It checks:

* Python
* PyTorch
* TorchAudio
* TorchVision
* CUDA
* PyWorld
* GPU
* VRAM
* Google Drive

The tested environment uses the known-good Beatrice configuration.

---

### 2. Project & Dataset

The user only needs to prepare a Google Drive folder named:

```text
Beatrice
```

Inside it, the user can place their training project and voice dataset ZIP.

The wrapper automatically searches for the relevant files and prepares the local Colab workspace.

There is no requirement for the dataset to be named `english`.

There is no requirement for a specific number of speakers.

For example:

```text
Beatrice/
│
├── MyVoice/
│   └── voices.zip
│
├── AnotherVoice/
│   └── voices.zip
│
└── checkpoints/
```

A dataset may contain one speaker or multiple speakers.

The wrapper discovers the actual structure instead of assuming a fixed speaker list.

---

### 3. Normal Training Settings

Provides simple settings for users who do not want to deal with advanced machine-learning parameters.

Examples include:

* Fresh training
* Resume training
* Training steps
* Checkpoint interval
* Evaluation interval
* Warmup steps
* Batch size
* CPU workers
* Automatic mixed precision

Hardware information is detected automatically where possible.

---

### 4. Advanced Settings

For experienced users, advanced Beatrice training parameters can be inspected and adjusted.

The wrapper exposes configuration values without changing the underlying Beatrice training implementation.

Advanced users can therefore use the same underlying trainer while having a more convenient configuration interface.

---

### 5. Training & Watchdog

Starts the original Beatrice V2 training process.

The wrapper also provides a simple watchdog for:

* Training status
* Checkpoint detection
* Checkpoint backup
* GPU monitoring
* Training output

The watchdog is intended to protect the user's work when using temporary cloud runtimes such as Google Colab.

---

# Checkpoint management

Checkpoint management is one of the main conveniences provided by this wrapper.

During training, Beatrice can create checkpoint files and `paraphernalia_*` directories.

The wrapper follows a simple storage policy:

### Historical checkpoints

Numbered checkpoints are preserved.

For example:

```text
checkpoint_00000500.pt.gz
checkpoint_00001000.pt.gz
checkpoint_00001500.pt.gz
```

These are **not continuously overwritten**.

### Latest checkpoint

The wrapper maintains a single rolling:

```text
checkpoint_latest.pt.gz
```

This avoids creating unnecessary duplicate copies of large checkpoint files every few seconds.

### Paraphernalia

`paraphernalia_*` directories are preserved.

They are not treated as disposable temporary files because they represent useful training progress/output.

---

# Fresh training

When the user selects:

**Fresh Training**

the wrapper starts a new training run.

Existing checkpoints are not automatically used as the training starting point.

---

# Resume training

When the user selects:

**Resume Training**

the wrapper searches the selected project for the available checkpoint data and prepares the latest valid checkpoint for the Beatrice trainer.

This allows a Colab session to be interrupted without losing the user's previous training progress.

---

# What this project does NOT do

This project does **not**:

* Reimplement Beatrice V2
* Replace the Beatrice V2 training engine
* Claim authorship of Beatrice V2
* Claim ownership of the underlying Beatrice technology
* Claim ownership of the original training code
* Modify the Beatrice training algorithm
* Present the underlying Beatrice technology as our own
* Guarantee training quality or model quality

The wrapper simply automates and simplifies the surrounding workflow.

---

# Relationship to Beatrice V2

The architecture is intentionally simple:

```text
                 BEATRICE V2
              ORIGINAL TRAINER
                     │
                     │
                     ▼
        ┌─────────────────────────┐
        │ Beatrice V2 Simple      │
        │ Trainer                 │
        │                         │
        │ Environment setup       │
        │ Dataset preparation     │
        │ Configuration           │
        │ Checkpoint management   │
        │ Watchdog                │
        │ Simple UI               │
        └─────────────────────────┘
                     │
                     ▼
              Google Colab
                     │
                     ▼
                Google Drive
```

The **training engine remains Beatrice V2**.

This repository simply makes the journey around that engine easier.

---

# Why this exists

There are already excellent engineers who have spent significant time developing and maintaining Beatrice V2 and its training infrastructure.

We do not want to reinvent that work.

Instead, this project focuses on a different problem:

> **How can an ordinary user access an existing powerful trainer without first becoming a Python engineer?**

That is the problem this wrapper attempts to solve.

---

# Credits

This project would not exist without the work of the Beatrice V2 developers and contributors.

Please visit and support the original projects:

* **Beatrice V2 Trainer:** `w-okada/beatrice-trainer-colab`
* **Beatrice Trainer:** `fierce-cats/beatrice-trainer`

The original Beatrice V2 projects contain the actual training technology.

This repository is only a convenience layer around that technology.

---

# Important

Please read and respect the licenses and terms of the original Beatrice V2 projects and any dependencies used by the trainer.

This repository does not grant additional rights to the underlying Beatrice V2 software.

Users are responsible for ensuring that their training data, voices, recordings, and resulting models are used lawfully and with the necessary permissions.

---

# License

The license for this wrapper applies only to the original code contained in this repository.

It does **not** replace or modify the licenses of Beatrice V2, its original training code, or third-party dependencies.

Please refer to the respective upstream repositories for their licenses and terms.

---

## The idea in one sentence

> **We didn't build Beatrice V2 — we simply made its training journey easier to use.**
