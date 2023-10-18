from face_detection import face_detection
from tqdm import tqdm
import os
import shutil

path = "./images"
images_with_person = f"{path}/images_with_person"
images_without_person = f"{path}/images_without_person"

images = os.listdir(path)

os.makedirs(name=images_with_person, exist_ok=True)
os.makedirs(name=images_without_person, exist_ok=True)

pbar_images = tqdm(images, desc="Processing")

for image in pbar_images:
    if os.path.isdir(image):
        continue

    faces = face_detection(f"{path}/{image}")

    if (faces > 0):
        shutil.move(
            f"{path}/{image}",
            images_with_person
        )
    else:
        shutil.move(
            f"{path}/{image}",
            images_without_person
        )
