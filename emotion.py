import cv2
from deepface import DeepFace
import mediapipe as mp


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

last_dominant_emotion, last_filtered_emotions = "Unknown", {"happy": 0.0, "neutral": 0.0, "surprise": 0.0}
frame_count = 0

entrant = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    if frame_count % 5 == 0:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list) and len(result) > 0:
                last_dominant_emotion = result[0]['dominant_emotion']
                last_filtered_emotions = {k: result[0]['emotion'][k] for k in ['happy', 'neutral', 'surprise']}
        except Exception as e:
            print(f"DeepFace error: {type(e).__name__} - {str(e)}")


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)  # Blue dots


    cv2.putText(frame, f"Emotion: {last_dominant_emotion}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for i, (emotion, score) in enumerate(last_filtered_emotions.items()):
        cv2.putText(frame, f"{emotion}: {score:.2f}%", (20, 50 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()