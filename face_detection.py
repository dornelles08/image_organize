import face_recognition


def face_detection(imgPath):
    image = face_recognition.load_image_file(imgPath)
    face_locations = face_recognition.face_locations(image)

    return len(face_locations)


if (__name__ == "__main__"):
    print(face_detection("images/C6965EBA-A01C-4A19-895E-29AEBDDE748D_1_105_c.jpeg"))
