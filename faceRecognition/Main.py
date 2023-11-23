import cv2
import face_recognition

# Tanınacak kişilerin yüzlerini yükleyin ve öğrenin
image1 = face_recognition.load_image_file("kisi1.jpg")
image1_encoding = face_recognition.face_encodings(image1)[0]

image2 = face_recognition.load_image_file("kisi2.jpg")
image2_encoding = face_recognition.face_encodings(image2)[0]

known_faces = [
    image1_encoding,
    image2_encoding
]

# Video kaynağını başlatın (webcam)
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Yüzlerin konumlarını bulun
    face_locations = face_recognition.face_locations(frame)

    # Yüzleri kodlayın
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        if True in matches:
            first_match_index = matches.index(True)
            if first_match_index == 0:
                name = "Kişi 1"
            elif first_match_index == 1:
                name = "Kişi 2"

            # Yüzün etrafına bir kutu ve isim çizin
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Sonuçları gösterin
    cv2.imshow('Video', frame)

    # 'q' tuşuna basıldığında döngüyü kırın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
