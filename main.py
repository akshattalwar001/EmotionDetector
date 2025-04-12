import cv2
from deepface import DeepFace


print(f"OpenCV version: {cv2.__version__}")


for index in range(3):  # Test indices 0, 1, 2
    print(f"Trying webcam index {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Webcam opened successfully at index {index}")
        break
    else:
        print(f"Failed to open webcam at index {index}")
        cap.release()


if not cap.isOpened():
    print("Error: No webcam could be opened. Check connection or drivers.")
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
print(f"Resolution set to: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")


last_dominant_emotion = "Unknown"
last_filtered_emotions = {"happy": 0.0, "neutral": 0.0, "surprise": 0.0}
frame_count = 0

while True:

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Webcam may be disconnected or inaccessible.")
        break

    frame_count += 1


    if frame_count % 5 == 0:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                last_dominant_emotion = first_result['dominant_emotion']
                emotion_scores = first_result['emotion']
                last_filtered_emotions = {key: emotion_scores[key] for key in ['happy', 'neutral', 'surprise'] if key in emotion_scores}
            else:
                print("No faces detected.")
        except Exception as e:
            print(f"DeepFace error: {type(e).__name__} - {str(e)}")


    cv2.putText(frame, f"Emotion: {last_dominant_emotion}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    for i, (emotion, score) in enumerate(last_filtered_emotions.items()):
        text = f"{emotion}: {score:.2f}%"
        cv2.putText(frame, text, (20, 50 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('Emotion Detector', frame)


    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        print("Exiting on user request ('q' pressed)")
        break


print("Releasing webcam and closing windows...")
cap.release()
cv2.destroyAllWindows()