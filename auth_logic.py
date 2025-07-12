import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

def carregar_usuarios(path="user_db.json"):
    with open(path, "r") as f:
        data = json.load(f)
        for user in data:
            user["face"] = np.array(user["face"])
            user["voice"] = np.array(user["voice"])
        return data

def autenticar(face_encoding, voice_embedding, usuarios):
    for user in usuarios:
        face_sim = 1 - np.linalg.norm(face_encoding - user["face"])
        voice_sim = cosine_similarity([voice_embedding], [user["voice"]])[0][0]

        print(f"[DEBUG] Face: {face_sim:.2f}, Voice: {voice_sim:.2f}")

        if face_sim > 0.55 and voice_sim > 0.70 and user["autorizado"]:
            return {"autenticado": True, "usuario": user["nome"]}
    return {"autenticado": False}
