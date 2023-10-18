import cv2


def face_detection(imgPath):
    img = cv2.imread(imgPath)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    print(face)

    # return len(face)

    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow("Image", img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if(__name__ == "__main__"):
    print(face_detection("images/3079.jpg"))