import os
import shutil

from tqdm import tqdm

from person_detection import person_detection

path = "./images"
images_with_person = f"{path}/images_with_person"
images_without_person = f"{path}/images_without_person"

images = os.listdir(path)

print(images)

os.makedirs(name=images_with_person, exist_ok=True)
os.makedirs(name=images_without_person, exist_ok=True)

pbar_images = tqdm(images, desc="Processing")

for image in pbar_images:
    try:
        if os.path.isdir(image):
            continue

        people = person_detection(f"{path}/{image}")

        if (people > 0):
            shutil.move(
                f"{path}/{image}",
                images_with_person
            )
        else:
            shutil.move(
                f"{path}/{image}",
                images_without_person
            )
    except Exception as e:
        print(e)
