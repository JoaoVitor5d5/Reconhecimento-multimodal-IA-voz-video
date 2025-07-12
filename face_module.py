import face_recognition
import cv2
import numpy as np

def capturar_face():
    video = cv2.VideoCapture(0)
    print("[INFO] Olhe para a câmera... Pressione 'q' para sair.")

    face_encoding = None
    while True:
        ret, frame = video.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) > 0:
            # Pegue a primeira face encontrada
            top, right, bottom, left = face_locations[0]
            face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[0]])

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                print("[INFO] Face detectada com sucesso!")
                break

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Cancelado pelo usuário.")
            break

    video.release()
    cv2.destroyAllWindows()
    return face_encoding
