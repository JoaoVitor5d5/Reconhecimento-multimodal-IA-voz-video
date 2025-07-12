import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from resemblyzer import VoiceEncoder, preprocess_wav
import os

def gravar_audio(duracao=5, fs=16000, nome_arquivo="audio.wav"):
    print(f"[INFO] Gravando áudio por {duracao} segundos...")
    audio = sd.rec(int(duracao * fs), samplerate=fs, channels=1)
    sd.wait()
    write(nome_arquivo, fs, audio)
    return nome_arquivo

def extrair_embedding(nome_arquivo="audio.wav"):
    wav = preprocess_wav(nome_arquivo)
    encoder = VoiceEncoder()
    emb = encoder.embed_utterance(wav)
    os.remove(nome_arquivo)
    return emb
