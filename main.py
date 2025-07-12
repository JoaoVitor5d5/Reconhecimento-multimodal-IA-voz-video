from face_module import capturar_face
from voice_module import gravar_audio, extrair_embedding
from auth_logic import carregar_usuarios, autenticar

if __name__ == "__main__":
    usuarios = carregar_usuarios()

    face_enc = capturar_face()
    if face_enc is None:
        print("[ERRO] Nenhum rosto detectado.")
        exit()

    audio_path = gravar_audio()
    voice_emb = extrair_embedding(audio_path)

    resultado = autenticar(face_enc, voice_emb, usuarios)

    if resultado["autenticado"]:
        print(f"[SUCESSO] Usuário autenticado: {resultado['usuario']}")
    else:
        print("[FALHA] Usuário não autenticado.")
