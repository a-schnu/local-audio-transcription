# Local-audio-transcription
A 100% local, VRAM-optimized pipeline for transcription + speaker diarization using Faster-Whisper and Pyannote. Built for 4GB GPUs but works everywhere!

# 🎙️ Local Whisper + Pyannote Diarization (Optimized for 4GB GPUs)

A completely local, highly optimized Python pipeline that combines **Faster-Whisper** for text transcription and **Pyannote** for speaker diarization. 

This script was specifically designed to solve the **Out Of Memory (OOM) bottleneck on 4GB VRAM GPUs** (like the RTX 3050). It achieves this by staging the AI models: it forces Whisper to use the GPU first, clears the VRAM, and then pushes Pyannote to the CPU (or GPU, if space permits).

## ✨ Key Features
* **VRAM Safe:** Automatically profiles your hardware and prevents medium/large models from crashing your 4GB GPU.
* **Staged Processing:** Transcription and Diarization run sequentially, ensuring they don't fight for the same memory budget.
* **Clean CLI:** Suppresses massive PyTorch, Triton, and Hugging Face warning walls for a clean terminal experience.
* **Smart Audio Preprocessing:** Uses PyAV and `numpy` for mono-mixdown, high-pass filtering, and energy-based VAD (Voice Activity Detection) before hitting the AI models.

---

## 🛠️ Prerequisites & Installation

### 1. System Requirements
Before running the script, ensure you have the following installed on your system:
* **Python 3.9+**
* **[FFmpeg](https://ffmpeg.org/download.html)**: Required by PyAV to decode audio files. (Make sure it's added to your system's PATH).
* **[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)**: Required for GPU acceleration.

### 2. Install Python Dependencies
It is highly recommended to use a virtual environment (`venv`). Activate it and run:

```bash
# Install standard dependencies
pip install faster-whisper pyannote.audio av numpy

# Install PyTorch with CUDA support (Crucial for GPU usage)
# Check [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for your specific CUDA version
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

🔑 Setup: Hugging Face & The "Fake Directory" Bypass
Pyannote requires you to accept their user conditions on Hugging Face to use their diarization models.

1. Get your Token:

Go to Hugging Face Settings and create a "Read" token.

Accept the user conditions for pyannote/speaker-diarization-3.1 (or community-1).

2. The Config Bypass:
Older versions of this script required a strict physical path to a downloaded config.yaml (a "fake local directory"). This script has been updated to bypass that! Simply pass the Hugging Face model ID and use the --allow-online-model-resolution flag. The script will use your token to verify, download the weights into your local cache once, and run locally forever after.

3. Export your Token:
Before running the script, load your token into your terminal environment:

Linux/macOS/MINGW: export HF_TOKEN="your_hugging_face_token_here"

Windows CMD: set HF_TOKEN=your_hugging_face_token_here

Windows PowerShell: $env:HF_TOKEN="your_hugging_face_token_here"

🚀 Usage
Here is the standard command to run the pipeline.

```bash
python interview_transcriber_diarized.py \
  "path/to/your/audio_file.wav" \
  --diarization-config "pyannote/speaker-diarization-community-1" \
  --whisper-model "small" \
  --transcription-device cuda \
  --diarization-device cpu \
  --allow-online-model-resolution
```
Pro-Tips for Model Selection:
Need more speed? Change --whisper-model "small" to "tiny".

Got an 8GB+ GPU? You can change --diarization-device cpu to cuda and upgrade Whisper to "turbo" or "large-v3".

📂 Output Files
The script doesn't just throw text at you; it builds a complete audit trail for your transcription:

1. _transcript.txt: The final, human-readable transcript with speaker labels and timestamps.

2. .rttm: Rich Transcription Time Marked. The universal standard file for speaker mapping, perfect for importing into video editors (like DaVinci Resolve) or research tools (like ELAN).

3. _report.json: The complete audit trail. Records exactly how long the pipeline took (RTF), which hardware was used, and the memory profile.

4. transcriber_state.sqlite: A persistent memory database. If you map "Speaker 1" to a real name, the SQLite DB remembers this for future runs!
