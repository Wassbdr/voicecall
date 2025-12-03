#!/usr/bin/env python3
"""
Agent Vocal IA - Version complete avec toutes les ameliorations.
- Clonage vocal ameliore (filtrage bruit, scoring qualite)
- Commandes vocales (stop, efface, personnalite)
- Wake word ("Hey Agent")
- Indicateur niveau micro
- Export conversation
- Logs fichier
- Reconnexion auto Ollama
"""

import os
import sys
import time
import wave
import json
import tempfile
import asyncio
import threading
import hashlib
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings("ignore")

# Configuration
OLLAMA_MODEL = "llama3.2:1b"
WHISPER_MODEL = "tiny"
MIN_SAMPLES = 2
OPTIMAL_SAMPLES = 5
MAX_SAMPLES = 15
SAMPLE_RATE = 22050
MIC_RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.2
MAX_HISTORY = 10
CACHE_SIZE = 50
AUTO_SAMPLE_MIN_DURATION = 2.5
AUTO_SAMPLE_MAX_DURATION = 8.0
WAKE_WORD = "hey agent"
LOG_FILE = "agent.log"

VOICE_COMMANDS = {
    "stop": ["stop", "arrete", "arrête", "tais-toi", "silence"],
    "clear": ["efface", "oublie", "reset", "recommence"],
    "ami": ["mode ami", "sois mon ami", "parle comme un ami"],
    "expert": ["mode expert", "sois expert"],
    "humour": ["mode humour", "sois drole", "sois drôle"],
    "assistant": ["mode assistant", "mode normal", "sois pro"],
}

PERSONALITIES = {
    "assistant": """Tu es un assistant vocal professionnel et efficace.
Tu reponds en francais de maniere claire et concise (2-3 phrases max).
Tu es poli mais direct, sans fioritures inutiles.""",
    
    "ami": """Tu es un ami proche qui discute de maniere decontractee.
Tu reponds en francais familier mais respectueux.
Tu es chaleureux, tu poses des questions, tu t'interesses a l'autre.""",
    
    "expert": """Tu es un expert technique patient et pedagogue.
Tu expliques les choses simplement en francais.
Tu donnes des exemples concrets quand c'est utile.""",
    
    "humour": """Tu es un assistant avec un sens de l'humour subtil.
Tu reponds en francais avec legerete et parfois un trait d'esprit.
Tu restes utile tout en etant agreable."""
}

# Logging fichier + console
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log(level: str, msg: str):
    level_map = {"INFO": logging.INFO, "OK": logging.INFO, "WARN": logging.WARNING, 
                 "ERR": logging.ERROR, "AUTO": logging.INFO, "USER": logging.INFO,
                 "IA": logging.INFO, "CMD": logging.INFO, "AUDIO": logging.INFO}
    logger.log(level_map.get(level, logging.INFO), f"[{level}] {msg}")

log("INFO", "Chargement des modules...")

import gradio as gr
from faster_whisper import WhisperModel
import edge_tts
import pygame
import pyaudio
import soundfile as sf
import librosa

# Ollama avec reconnexion
import ollama
ollama_connected = False

def check_ollama():
    global ollama_connected
    try:
        ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "ok"}])
        ollama_connected = True
        return True
    except:
        ollama_connected = False
        return False

log("INFO", "Chargement OpenVoice...")
try:
    from openvoice_cli import se_extractor
    from openvoice_cli.api import ToneColorConverter
    import torch
    OPENVOICE_OK = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    log("OK", f"OpenVoice charge (device: {DEVICE})")
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).parent / "OpenVoice"))
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        import torch
        OPENVOICE_OK = True
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        log("OK", f"OpenVoice charge (device: {DEVICE})")
    except ImportError as e:
        OPENVOICE_OK = False
        DEVICE = "cpu"
        log("WARN", f"OpenVoice non disponible: {e}")

log("INFO", "Verification Ollama...")
if check_ollama():
    log("OK", "Ollama connecte")
else:
    log("WARN", "Ollama non disponible - lancez: ollama serve")

log("INFO", f"Chargement Whisper ({WHISPER_MODEL})...")
whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
log("OK", "Whisper pret")

pygame.mixer.init(frequency=SAMPLE_RATE)
log("OK", "Audio initialise")

SAMPLES_DIR = Path("voice_samples")
OUTPUT_DIR = Path("voice_clone_output")
EXPORT_DIR = Path("exports")
CKPT_DIR = Path("openvoice_model/checkpoints")
for d in [SAMPLES_DIR, OUTPUT_DIR, EXPORT_DIR]:
    d.mkdir(exist_ok=True)


class ResponseCache:
    def __init__(self, max_size: int = CACHE_SIZE):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
    
    def _key(self, text: str) -> str:
        normalized = " ".join(text.lower().strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def get(self, text: str) -> Optional[str]:
        key = self._key(text)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, text: str, response: str):
        key = self._key(text)
        self.cache[key] = response
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self):
        self.cache.clear()


class AudioAnalyzer:
    """Analyse qualite audio pour filtrage."""
    
    @staticmethod
    def compute_snr(audio: np.ndarray, sr: int) -> float:
        """Estime le rapport signal/bruit."""
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)
        
        frames = []
        for i in range(0, len(audio) - frame_len, hop):
            frames.append(np.abs(audio[i:i+frame_len]).mean())
        
        if not frames:
            return 0.0
        
        frames = np.array(frames)
        noise_floor = np.percentile(frames, 10)
        signal_level = np.percentile(frames, 90)
        
        if noise_floor < 1e-6:
            noise_floor = 1e-6
        
        snr = 20 * np.log10(signal_level / noise_floor)
        return max(0, min(60, snr))
    
    @staticmethod
    def compute_clarity(audio: np.ndarray, sr: int) -> float:
        """Estime la clarte vocale (0-100)."""
        try:
            S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            
            voice_mask = (freqs >= 100) & (freqs <= 4000)
            voice_energy = S[voice_mask].sum()
            total_energy = S.sum() + 1e-10
            
            clarity = (voice_energy / total_energy) * 100
            return min(100, clarity)
        except:
            return 50.0
    
    @staticmethod
    def score_sample(audio: np.ndarray, sr: int) -> Tuple[float, str]:
        """Score global d'un echantillon (0-100) avec feedback."""
        snr = AudioAnalyzer.compute_snr(audio, sr)
        clarity = AudioAnalyzer.compute_clarity(audio, sr)
        
        duration = len(audio) / sr
        duration_score = min(100, max(0, (duration - 1) * 20))
        
        peak = np.abs(audio).max()
        level_score = min(100, peak * 150) if peak < 0.9 else 80
        
        score = (snr * 0.3 + clarity * 0.3 + duration_score * 0.2 + level_score * 0.2)
        
        if score >= 70:
            feedback = "Excellent"
        elif score >= 50:
            feedback = "Bon"
        elif score >= 30:
            feedback = "Acceptable"
        else:
            feedback = "Faible qualite"
        
        return score, feedback
    
    @staticmethod
    def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
        """Reduction de bruit simple."""
        try:
            from scipy import signal as sig
            
            # Filtre passe-haut (coupe < 80Hz)
            b, a = sig.butter(4, 80 / (sr / 2), btype='high')
            audio = sig.filtfilt(b, a, audio)
            
            # Filtre passe-bas (coupe > 8000Hz)
            b, a = sig.butter(4, 8000 / (sr / 2), btype='low')
            audio = sig.filtfilt(b, a, audio)
            
            # Gate simple
            threshold = np.percentile(np.abs(audio), 20)
            mask = np.abs(audio) > threshold
            audio = audio * (mask * 0.9 + 0.1)
            
            return audio
        except:
            return audio


class VoiceCloner:
    def __init__(self):
        self.samples: List[Dict] = []  # {path, score, duration}
        self.ready = False
        self.preparing = False
        self.converter = None
        self.target_se = None
        self.source_se = None
        self.quality = 0.0
        self._callback = None
        self._last_extract_count = 0
        self.auto_sample = True
        self.denoise_enabled = True
        
        self._init_model()
        self._load_samples()
    
    def _init_model(self):
        if not OPENVOICE_OK:
            return
        
        try:
            config = CKPT_DIR / "converter" / "config.json"
            if not config.exists():
                log("WARN", f"Config manquante: {config}")
                return
            
            self.converter = ToneColorConverter(str(config), device=DEVICE)
            self.converter.load_ckpt(str(CKPT_DIR / "converter" / "checkpoint.pth"))
            
            source_path = CKPT_DIR / "base_speakers" / "EN" / "en_default_se.pth"
            if source_path.exists():
                self.source_se = torch.load(source_path, map_location=DEVICE)
            
            log("OK", "OpenVoice initialise")
        except Exception as e:
            log("ERR", f"Erreur OpenVoice: {e}")
    
    def _load_samples(self):
        for path in sorted(SAMPLES_DIR.glob("*.wav")):
            try:
                audio, sr = librosa.load(str(path), sr=SAMPLE_RATE)
                score, _ = AudioAnalyzer.score_sample(audio, sr)
                self.samples.append({
                    "path": path,
                    "score": score,
                    "duration": len(audio) / sr
                })
            except:
                pass
        
        log("INFO", f"{len(self.samples)} echantillons charges")
        
        se_path = OUTPUT_DIR / "target_se.pth"
        if se_path.exists() and OPENVOICE_OK:
            try:
                self.target_se = torch.load(se_path, map_location=DEVICE)
                self.ready = True
                self._update_quality()
                log("OK", "Profil vocal charge")
            except Exception as e:
                log("WARN", f"Erreur chargement profil: {e}")
        
        if len(self.samples) >= MIN_SAMPLES and not self.ready:
            threading.Thread(target=self._extract, daemon=True).start()
    
    def _update_quality(self):
        if not self.samples:
            self.quality = 0.0
            return
        
        avg_score = sum(s["score"] for s in self.samples) / len(self.samples)
        count_factor = min(1.0, len(self.samples) / OPTIMAL_SAMPLES)
        
        self.quality = avg_score * count_factor
    
    def set_callback(self, fn):
        self._callback = fn
    
    def add(self, audio_data: tuple) -> str:
        if audio_data is None:
            return "Erreur: pas d'audio"
        
        try:
            sr, data = audio_data
            
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            duration = len(data) / sr
            if duration < 2.0:
                return f"Erreur: trop court ({duration:.1f}s < 2s)"
            
            if self.denoise_enabled:
                data = AudioAnalyzer.denoise(data, sr)
            
            if sr != SAMPLE_RATE:
                data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
            
            peak = np.abs(data).max()
            if peak > 0:
                data = data / peak * 0.9
            
            score, feedback = AudioAnalyzer.score_sample(data, SAMPLE_RATE)
            
            if score < 25:
                return f"Qualite insuffisante ({feedback}, score: {score:.0f})"
            
            path = SAMPLES_DIR / f"sample_{len(self.samples)+1:02d}.wav"
            sf.write(str(path), data, SAMPLE_RATE)
            
            self.samples.append({"path": path, "score": score, "duration": duration})
            self._update_quality()
            
            msg = f"Ajoute: {feedback} (score: {score:.0f}, {duration:.1f}s)"
            
            if len(self.samples) >= MIN_SAMPLES and not self.ready and not self.preparing:
                msg += " - Extraction..."
                threading.Thread(target=self._extract, daemon=True).start()
            
            return msg
        except Exception as e:
            return f"Erreur: {e}"
    
    def add_from_conversation(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        if not self.auto_sample:
            return False
        
        try:
            duration = len(audio) / sample_rate
            if duration < AUTO_SAMPLE_MIN_DURATION or duration > AUTO_SAMPLE_MAX_DURATION:
                return False
            
            data = audio.astype(np.float32) / 32768.0
            
            energy = np.abs(data).mean()
            if energy < 0.01:
                return False
            
            if self.denoise_enabled:
                data = AudioAnalyzer.denoise(data, sample_rate)
            
            if sample_rate != SAMPLE_RATE:
                data = librosa.resample(data, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
            
            peak = np.abs(data).max()
            if peak > 0:
                data = data / peak * 0.9
            
            score, feedback = AudioAnalyzer.score_sample(data, SAMPLE_RATE)
            
            if score < 30:
                return False
            
            # Gestion max samples - remplace le moins bon
            if len(self.samples) >= MAX_SAMPLES:
                worst = min(self.samples, key=lambda s: s["score"])
                if score <= worst["score"]:
                    return False
                worst["path"].unlink()
                self.samples.remove(worst)
                log("AUTO", f"Remplace {worst['path'].name} (score: {worst['score']:.0f})")
            
            idx = len(list(SAMPLES_DIR.glob("auto_*.wav"))) + 1
            path = SAMPLES_DIR / f"auto_{idx:03d}.wav"
            sf.write(str(path), data, SAMPLE_RATE)
            
            self.samples.append({"path": path, "score": score, "duration": duration})
            self._update_quality()
            
            log("AUTO", f"Echantillon ({feedback}, score: {score:.0f}, {duration:.1f}s)")
            
            if len(self.samples) >= MIN_SAMPLES and not self.preparing:
                new_samples = len(self.samples) - self._last_extract_count
                if not self.ready or new_samples >= 3:
                    threading.Thread(target=self._extract, daemon=True).start()
            
            return True
        except Exception as e:
            log("WARN", f"Auto-sample: {e}")
            return False
    
    def _extract(self):
        if not OPENVOICE_OK or not self.converter:
            return
        
        self.preparing = True
        self._last_extract_count = len(self.samples)
        log("INFO", "Extraction profil vocal...")
        
        try:
            # Trie par score et prend les meilleurs
            sorted_samples = sorted(self.samples, key=lambda s: s["score"], reverse=True)
            best_samples = sorted_samples[:min(10, len(sorted_samples))]
            
            chunks = []
            for sample in best_samples:
                audio, _ = librosa.load(str(sample["path"]), sr=SAMPLE_RATE)
                if self.denoise_enabled:
                    audio = AudioAnalyzer.denoise(audio, SAMPLE_RATE)
                chunks.append(audio)
                chunks.append(np.zeros(int(SAMPLE_RATE * 0.3)))
            
            combined = np.concatenate(chunks)
            
            # Normalisation finale
            peak = np.abs(combined).max()
            if peak > 0:
                combined = combined / peak * 0.9
            
            ref_path = OUTPUT_DIR / "reference_voice.wav"
            sf.write(str(ref_path), combined, SAMPLE_RATE)
            
            result = se_extractor.get_se(str(ref_path), self.converter, vad=True)
            self.target_se = result[0] if isinstance(result, tuple) else result
            
            torch.save(self.target_se, OUTPUT_DIR / "target_se.pth")
            
            self.ready = True
            self._update_quality()
            log("OK", f"Profil cree (qualite: {self.quality:.0f}%)")
            
            if self._callback:
                self._callback()
            
        except Exception as e:
            log("ERR", f"Extraction: {e}")
        finally:
            self.preparing = False
    
    def convert(self, src_path: str, out_path: str) -> bool:
        if not self.ready or self.target_se is None or self.source_se is None:
            return False
        
        try:
            self.converter.convert(
                audio_src_path=src_path,
                src_se=self.source_se,
                tgt_se=self.target_se,
                output_path=out_path,
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                return True
            return False
        except Exception as e:
            log("WARN", f"Conversion: {e}")
            return False
    
    def status(self) -> str:
        lines = [f"Echantillons: {len(self.samples)}/{MAX_SAMPLES}"]
        
        if self.samples:
            avg_score = sum(s["score"] for s in self.samples) / len(self.samples)
            lines.append(f"Score moyen: {avg_score:.0f}/100")
        
        if self.ready:
            lines.append(f"Pret - Qualite: {self.quality:.0f}%")
        elif self.preparing:
            lines.append("Preparation...")
        elif len(self.samples) < MIN_SAMPLES:
            lines.append(f"Manque {MIN_SAMPLES - len(self.samples)} echantillon(s)")
        
        opts = []
        if self.auto_sample:
            opts.append("Auto")
        if self.denoise_enabled:
            opts.append("Denoise")
        lines.append(f"Options: {', '.join(opts) if opts else 'Aucune'}")
        
        return "\n".join(lines)
    
    def reset(self):
        for s in self.samples:
            try:
                s["path"].unlink()
            except:
                pass
        for f in OUTPUT_DIR.glob("*"):
            if f.is_file():
                f.unlink()
        self.samples = []
        self.ready = False
        self.target_se = None
        self.quality = 0.0
        return "Donnees supprimees"


class Agent:
    def __init__(self, cloner: VoiceCloner):
        self.cloner = cloner
        self.listening = False
        self.speaking = False
        self.processing = False
        self.use_clone = False
        self.personality = "assistant"
        self.wake_word_enabled = False
        self.wake_word_active = True
        
        self.history: List[Dict] = []
        self.cache = ResponseCache()
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.buffer = []
        self.silence_count = 0
        self.mic_level = 0.0
        
        self.state = "idle"
        self.errors = 0
        self.last_command = ""
        
        cloner.set_callback(self._on_clone_ready)
        if cloner.ready:
            self.use_clone = True
    
    def _on_clone_ready(self):
        self.use_clone = True
        log("AUTO", "Clonage active")
    
    def _check_voice_command(self, text: str) -> Optional[str]:
        """Verifie si le texte est une commande vocale."""
        text_lower = text.lower().strip()
        
        for cmd, triggers in VOICE_COMMANDS.items():
            for trigger in triggers:
                if trigger in text_lower:
                    return cmd
        return None
    
    def _execute_command(self, cmd: str) -> str:
        """Execute une commande vocale."""
        self.last_command = cmd
        
        if cmd == "stop":
            self.interrupt()
            return "D'accord, je me tais."
        elif cmd == "clear":
            self.clear()
            return "Conversation effacee."
        elif cmd in PERSONALITIES:
            self.personality = cmd
            return f"Mode {cmd} active."
        
        return ""
    
    def set_personality(self, name: str) -> str:
        if name in PERSONALITIES:
            self.personality = name
            return f"Personnalite: {name}"
        return f"Inconnue: {name}"
    
    def start(self) -> str:
        if self.listening:
            return "Deja en ecoute"
        
        self.listening = True
        self.buffer = []
        self.silence_count = 0
        self.state = "listening"
        self.errors = 0
        
        if self.wake_word_enabled:
            self.wake_word_active = False
        
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=MIC_RATE,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self._on_audio
            )
            self.stream.start_stream()
            return "Ecoute active"
        except Exception as e:
            self.listening = False
            self.state = "idle"
            return f"Erreur micro: {e}"
    
    def stop(self) -> str:
        self.listening = False
        self.state = "idle"
        self.mic_level = 0.0
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        return "Ecoute arretee"
    
    def interrupt(self) -> str:
        if self.speaking:
            pygame.mixer.music.stop()
            self.speaking = False
            self.state = "listening" if self.listening else "idle"
            return "Interrompu"
        return ""
    
    def _on_audio(self, data, frames, time_info, status):
        if not self.listening:
            return (None, pyaudio.paComplete)
        
        samples = np.frombuffer(data, dtype=np.int16)
        level = np.abs(samples).mean()
        self.mic_level = min(100, level / 200)
        
        if self.speaking and level > SILENCE_THRESHOLD * 2:
            self.interrupt()
            return (None, pyaudio.paContinue)
        
        if self.speaking or self.processing:
            return (None, pyaudio.paContinue)
        
        if level > SILENCE_THRESHOLD:
            self.buffer.append(samples)
            self.silence_count = 0
        elif self.buffer:
            self.buffer.append(samples)
            self.silence_count += 1
            
            if self.silence_count >= int(SILENCE_DURATION * MIC_RATE / 1024):
                audio = np.concatenate(self.buffer)
                self.buffer = []
                self.silence_count = 0
                threading.Thread(target=self._process, args=(audio,), daemon=True).start()
        
        return (None, pyaudio.paContinue)
    
    def _process(self, audio: np.ndarray):
        if self.processing:
            return
        
        self.processing = True
        self.state = "processing"
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            
            with wave.open(path, 'w') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(MIC_RATE)
                wav.writeframes(audio.tobytes())
            
            segments, _ = whisper_model.transcribe(path, language="fr")
            text = " ".join([s.text for s in segments]).strip()
            os.unlink(path)
            
            if len(text) < 2:
                self.processing = False
                self.state = "listening"
                return
            
            # Wake word check
            if self.wake_word_enabled and not self.wake_word_active:
                if WAKE_WORD in text.lower():
                    self.wake_word_active = True
                    log("CMD", "Wake word detecte")
                    self._speak("Oui, je t'ecoute.")
                self.processing = False
                self.state = "listening"
                return
            
            log("USER", text)
            
            # Check commandes vocales
            cmd = self._check_voice_command(text)
            if cmd:
                log("CMD", f"Commande: {cmd}")
                response = self._execute_command(cmd)
                if response:
                    self._speak(response)
                self.processing = False
                self.state = "listening"
                return
            
            # Auto-sampling
            self.cloner.add_from_conversation(audio, MIC_RATE)
            
            # Reponse
            if not check_ollama():
                log("WARN", "Ollama deconnecte, tentative reconnexion...")
                time.sleep(1)
                if not check_ollama():
                    self._speak("Desolee, je ne peux pas repondre pour le moment.")
                    self.processing = False
                    self.state = "listening"
                    return
            
            response = self.cache.get(text)
            if response:
                log("CACHE", "Cache hit")
            else:
                response = self._respond(text)
                self.cache.put(text, response)
            
            log("IA", response)
            self._speak(response)
            self.errors = 0
            
        except Exception as e:
            log("ERR", f"Traitement: {e}")
            self.errors += 1
            if self.errors >= 3:
                self.stop()
        finally:
            self.processing = False
            self.state = "listening" if self.listening else "idle"
    
    def _respond(self, text: str) -> str:
        prompt = PERSONALITIES.get(self.personality, PERSONALITIES["assistant"])
        
        messages = [{"role": "system", "content": prompt}]
        messages.extend(self.history[-MAX_HISTORY:])
        messages.append({"role": "user", "content": text})
        
        try:
            result = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={"num_predict": 150, "temperature": 0.7}
            )
            
            reply = result['message']['content']
            
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
            
            if len(self.history) > MAX_HISTORY * 2:
                self.history = self.history[-MAX_HISTORY * 2:]
            
            return reply
        except Exception as e:
            return f"Erreur: {e}"
    
    def _speak(self, text: str):
        self.speaking = True
        self.state = "speaking"
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def synth():
                comm = edge_tts.Communicate(text, "fr-FR-DeniseNeural")
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    path = f.name
                await comm.save(path)
                return path
            
            src = loop.run_until_complete(synth())
            loop.close()
            
            out = src
            clone_ok = False
            wav_src = None
            wav_out = None
            
            if self.use_clone and self.cloner.ready:
                try:
                    audio, _ = librosa.load(src, sr=SAMPLE_RATE)
                    wav_src = src.replace(".mp3", "_src.wav")
                    sf.write(wav_src, audio, SAMPLE_RATE)
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        wav_out = f.name
                    
                    if self.cloner.convert(wav_src, wav_out):
                        os.unlink(src)
                        os.unlink(wav_src)
                        out = wav_out
                        clone_ok = True
                    else:
                        os.unlink(wav_src)
                        try:
                            os.unlink(wav_out)
                        except:
                            pass
                except Exception as e:
                    log("WARN", f"Clone fallback: {e}")
                    for f in [wav_src, wav_out]:
                        if f:
                            try:
                                os.unlink(f)
                            except:
                                pass
            
            if not self.speaking:
                try:
                    os.unlink(out)
                except:
                    pass
                return
            
            pygame.mixer.music.load(out)
            pygame.mixer.music.play()
            log("AUDIO", f"{'Clone' if clone_ok else 'Standard'}")
            
            while pygame.mixer.music.get_busy() and self.speaking:
                time.sleep(0.05)
            
            try:
                os.unlink(out)
            except:
                pass
            
        except Exception as e:
            log("ERR", f"TTS: {e}")
        finally:
            self.speaking = False
            self.state = "listening" if self.listening else "idle"
    
    def get_history(self) -> str:
        if not self.history:
            return "*En attente...*"
        
        lines = []
        for msg in self.history[-10:]:
            role = "Vous" if msg["role"] == "user" else "Agent"
            lines.append(f"**{role}:** {msg['content']}")
        return "\n\n".join(lines)
    
    def get_state(self) -> str:
        states = {
            "idle": "Pause", 
            "listening": "Ecoute", 
            "processing": "Reflexion", 
            "speaking": "Parle"
        }
        return states.get(self.state, "?")
    
    def get_mic_level(self) -> float:
        return self.mic_level
    
    def clear(self):
        self.history = []
        self.cache.clear()
    
    def export_conversation(self) -> str:
        if not self.history:
            return "Rien a exporter"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON
        json_path = EXPORT_DIR / f"conversation_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "personality": self.personality,
                "messages": self.history
            }, f, ensure_ascii=False, indent=2)
        
        # TXT
        txt_path = EXPORT_DIR / f"conversation_{timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Conversation - {timestamp}\n")
            f.write(f"Personnalite: {self.personality}\n")
            f.write("=" * 50 + "\n\n")
            for msg in self.history:
                role = "Vous" if msg["role"] == "user" else "Agent"
                f.write(f"{role}: {msg['content']}\n\n")
        
        return f"Exporte: {json_path.name}"


# Instances
log("INFO", "Initialisation...")
cloner = VoiceCloner()
agent = Agent(cloner)


# Interface helpers
def on_listen(active: bool) -> str:
    return agent.start() if active else agent.stop()

def get_status() -> str:
    state = agent.get_state()
    info = []
    if agent.use_clone and cloner.ready:
        info.append("Clone")
    if agent.wake_word_enabled:
        info.append("Wake" + (" ON" if agent.wake_word_active else " OFF"))
    extra = f" | {', '.join(info)}" if info else ""
    return f"**{state}**{extra}\n\n{cloner.status()}"

def get_mic_html() -> str:
    level = agent.get_mic_level()
    color = "#4CAF50" if level > 30 else "#FFC107" if level > 10 else "#666"
    return f'''
    <div style="background:#222; border-radius:8px; padding:10px; text-align:center;">
        <div style="background:#333; border-radius:4px; height:20px; overflow:hidden;">
            <div style="background:{color}; height:100%; width:{level}%; transition:width 0.1s;"></div>
        </div>
        <small style="color:#888;">Niveau: {level:.0f}%</small>
    </div>
    '''


log("INFO", "Creation interface...")

with gr.Blocks(title="Agent Vocal") as app:
    
    gr.HTML("""
    <div style="text-align:center; padding:20px; background:linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius:12px; margin-bottom:20px;">
        <h1 style="color:#eee; margin:0; font-size:2em;">Agent Vocal</h1>
        <p style="color:#888; margin:8px 0 0 0;">Ecoute continue | Clonage vocal | Commandes vocales</p>
    </div>
    """)
    
    with gr.Tabs():
        
        with gr.Tab("Conversation"):
            with gr.Row():
                with gr.Column(scale=1):
                    listen_cb = gr.Checkbox(label="Ecoute active", value=False)
                    clone_cb = gr.Checkbox(label="Voix clonee", value=agent.use_clone)
                    wake_cb = gr.Checkbox(label=f"Wake word ({WAKE_WORD})", value=False)
                    
                    mic_html = gr.HTML(value=get_mic_html())
                    
                    personality_dd = gr.Dropdown(
                        choices=list(PERSONALITIES.keys()), 
                        value="assistant", 
                        label="Personnalite"
                    )
                    
                    with gr.Row():
                        interrupt_btn = gr.Button("Stop", variant="stop", size="sm")
                        clear_btn = gr.Button("Effacer", size="sm")
                        export_btn = gr.Button("Export", size="sm")
                    
                    status_md = gr.Markdown(value=get_status())
                    export_msg = gr.Markdown()
                
                with gr.Column(scale=2):
                    gr.Markdown("### Historique")
                    history_md = gr.Markdown(value="*Activez l'ecoute pour commencer...*")
                    
                    gr.Markdown("""
                    ---
                    **Commandes vocales:** "stop/arrete", "efface/oublie", "mode ami/expert/humour"
                    """)
            
            listen_cb.change(on_listen, [listen_cb], [status_md])
            clone_cb.change(lambda x: setattr(agent, 'use_clone', x) or get_status(), [clone_cb], [status_md])
            wake_cb.change(lambda x: setattr(agent, 'wake_word_enabled', x) or get_status(), [wake_cb], [status_md])
            personality_dd.change(lambda x: agent.set_personality(x), [personality_dd], [status_md])
            interrupt_btn.click(agent.interrupt, outputs=[status_md])
            clear_btn.click(lambda: agent.clear() or "", outputs=[history_md])
            export_btn.click(agent.export_conversation, outputs=[export_msg])
            
            timer = gr.Timer(0.3, active=True)
            timer.tick(agent.get_history, outputs=[history_md])
            timer.tick(get_status, outputs=[status_md])
            timer.tick(get_mic_html, outputs=[mic_html])
            timer.tick(lambda: agent.use_clone, outputs=[clone_cb])
        
        with gr.Tab("Clonage"):
            gr.Markdown("""
            ### Clonage Vocal
            Enregistrez des echantillons de votre voix pour que l'agent parle avec votre voix.
            - **Minimum:** 2 echantillons de 2+ secondes
            - **Optimal:** 5-10 echantillons varies
            - **Auto-sampling:** Collecte automatique pendant la conversation
            """)
            
            with gr.Row():
                with gr.Column():
                    auto_cb = gr.Checkbox(label="Auto-sampling", value=cloner.auto_sample)
                    denoise_cb = gr.Checkbox(label="Reduction bruit", value=cloner.denoise_enabled)
                    
                    audio_in = gr.Audio(sources=["microphone"], label="Enregistrer (min 2s)")
                    add_btn = gr.Button("Ajouter echantillon", variant="primary")
                    result_md = gr.Markdown()
                
                with gr.Column():
                    clone_status_md = gr.Markdown(value=cloner.status())
                    
                    gr.Markdown("---")
                    reset_btn = gr.Button("Supprimer tout", variant="stop")
            
            auto_cb.change(lambda x: setattr(cloner, 'auto_sample', x) or cloner.status(), [auto_cb], [clone_status_md])
            denoise_cb.change(lambda x: setattr(cloner, 'denoise_enabled', x) or cloner.status(), [denoise_cb], [clone_status_md])
            add_btn.click(lambda a: (cloner.add(a), cloner.status()), [audio_in], [result_md, clone_status_md])
            reset_btn.click(lambda: (cloner.reset(), cloner.status()), outputs=[result_md, clone_status_md])
            
            timer2 = gr.Timer(2.0, active=True)
            timer2.tick(cloner.status, outputs=[clone_status_md])
        
        with gr.Tab("Configuration"):
            gr.Markdown(f"""
            ### Parametres systeme
            
            | Composant | Valeur |
            |-----------|--------|
            | **Whisper** | `{WHISPER_MODEL}` (CPU, int8) |
            | **Ollama** | `{OLLAMA_MODEL}` |
            | **OpenVoice** | `{'OK' if OPENVOICE_OK else 'Non'}` ({DEVICE}) |
            | **TTS** | Edge TTS (fr-FR-DeniseNeural) |
            
            ### Audio
            
            | Parametre | Valeur |
            |-----------|--------|
            | Sample rate | {SAMPLE_RATE} Hz |
            | Micro rate | {MIC_RATE} Hz |
            | Seuil silence | {SILENCE_THRESHOLD} |
            | Duree silence | {SILENCE_DURATION}s |
            
            ### Clonage
            
            | Parametre | Valeur |
            |-----------|--------|
            | Min echantillons | {MIN_SAMPLES} |
            | Max echantillons | {MAX_SAMPLES} |
            | Auto-sample duree | {AUTO_SAMPLE_MIN_DURATION}-{AUTO_SAMPLE_MAX_DURATION}s |
            
            ### Memoire
            
            | Parametre | Valeur |
            |-----------|--------|
            | Historique | {MAX_HISTORY} messages |
            | Cache | {CACHE_SIZE} reponses |
            
            ### Fichiers
            
            - **Logs:** `{LOG_FILE}`
            - **Exports:** `{EXPORT_DIR}/`
            - **Echantillons:** `{SAMPLES_DIR}/`
            """)


log("OK", f"Lancement http://localhost:7864")
app.launch(server_name="0.0.0.0", server_port=7864, share=False)
