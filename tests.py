import cv2
import face_recognition

felipe_img_path = "images/Captura de Tela 2023-10-18 às 15.51.27.png"
imgPath = "images/6 abril 2022.jpg"

image = face_recognition.load_image_file(imgPath)
face_locations = face_recognition.face_locations(image, model="cnn")
face_encoding = face_recognition.face_encodings(image)

felipe_image = face_recognition.load_image_file(felipe_img_path)
felipe_face_encoding = face_recognition.face_encodings(felipe_image)[0]

if face_encoding:
    matches = face_recognition.compare_faces(
        felipe_face_encoding, face_encoding)

    print(matches)

    face_distances = face_recognition.face_distance(
        felipe_face_encoding, face_encoding)
    print(face_distances)

img = cv2.imread(imgPath)

for (top, right, bottom, left) in face_locations:
    img = cv2.rectangle(img, (left, top), (right, bottom),
                        color=(255, 0, 0), thickness=2)

cv2.imshow("Image", img)
cv2.waitKey(0)

print(face_locations)
