# Local Voice Agent

A fully local voice assistant that combines speech-to-text, a large language model, text-to-speech, and voice cloning. It runs entirely on your machine, ensuring privacy and independence from external APIs.

## Architecture

The project integrates several open-source technologies to create a seamless voice interaction loop:

- **Speech-to-Text (STT):** Uses `faster-whisper` for fast and accurate transcription.
- **Large Language Model (LLM):** Connects to a local Ollama instance (e.g., Llama 3) for intelligence.
- **Text-to-Speech (TTS):** Uses Edge TTS for high-quality, natural-sounding speech synthesis.
- **Voice Cloning:** Integrates OpenVoice to clone voices with minimal latency.

## Features

- **Privacy-First:** No audio or text data leaves your machine.
- **Real-time Voice Cloning:** Can clone a voice from a reference file or learn from the user's voice during the conversation.
- **Smart Audio Analysis:** Evaluates input audio quality (SNR, clarity) to ensure only high-quality samples are used for cloning.
- **Wake Word Detection:** Activates on "Hey Agent".
- **Web Interface:** Provides a user-friendly interface via Gradio.

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/) installed and running
- FFmpeg installed on your system

## Installation

1.  **Install Ollama**
    Follow the instructions on the official website to install Ollama.
    Pull a model (e.g., Llama 3.2):
    ```bash
    ollama pull llama3.2:3b
    ```

2.  **Install Python Dependencies**
    Create a virtual environment and install the required packages:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **OpenVoice Setup**
    Ensure that the OpenVoice checkpoints are correctly placed in the `openvoice_model/checkpoints/` directory.

## Usage

1.  **Start Ollama**
    Make sure the Ollama server is running:
    ```bash
    ollama serve
    ```

2.  **Run the Agent**
    Start the application:
    ```bash
    python agent.py
    ```

3.  **Access the Interface**
    Open your web browser and navigate to the URL provided in the terminal (typically `http://localhost:7860`).

## Configuration

You can adjust various settings directly in `agent.py`, such as:
- `OLLAMA_MODEL`: The model used for generating responses.
- `SYSTEM_PROMPT`: The personality and instructions for the agent.
- Audio processing parameters (thresholds, silence detection).

## Authors

Alexandre Da Silva & Wassim Badraoui
