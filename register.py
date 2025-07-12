import os
import json
import numpy as np
from face_module import capturar_face
from voice_module import gravar_audio, extrair_embedding

DB_PATH = "user_db.json"

def salvar_usuario(nome, face_encoding, voice_embedding):
    # Converte os vetores numpy para listas (JSON não suporta numpy)
    novo_usuario = {
        "nome": nome,
        "face": face_encoding.tolist(),
        "voice": voice_embedding.tolist(),
        "autorizado": True
    }

    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            usuarios = json.load(f)
    else:
        usuarios = []

    usuarios.append(novo_usuario)

    with open(DB_PATH, "w") as f:
        json.dump(usuarios, f, indent=4)
    
    print(f"[SUCESSO] Usuário '{nome}' registrado com sucesso.")

if __name__ == "__main__":
    print("=== REGISTRO DE NOVO USUÁRIO ===")
    nome = input("Digite o nome do usuário: ").strip()

    face_enc = capturar_face()
    if face_enc is None:
        print("[ERRO] Nenhum rosto detectado.")
        exit()

    audio_path = gravar_audio()
    voice_emb = extrair_embedding(audio_path)

    salvar_usuario(nome, face_enc, voice_emb)
