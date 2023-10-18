import os
import shutil

from tqdm import tqdm
from tinydb import TinyDB

from person_detection import person_detection

database = TinyDB("database.json", indent=4)

path = "./images"
images_with_person = f"{path}/images_with_person"
images_without_person = f"{path}/images_without_person"

images = os.listdir(path)

print(len(images))

# os.makedirs(name=images_with_person, exist_ok=True)
# os.makedirs(name=images_without_person, exist_ok=True)

pbar_images = tqdm(images, desc="Processing")

for image in pbar_images:
    try:
        if os.path.isdir(f"{path}/{image}"):
            continue

        people = person_detection(f"{path}/{image}")

        if (people > 0):
            database.insert({
                "image": image,
                "hasPerson": True
            })
            # shutil.move(
            #     f"{path}/{image}",
            #     images_with_person
            # )
        else:
            database.insert({
                "image": image,
                "hasPerson": False
            })
            # shutil.move(
            #     f"{path}/{image}",
            #     images_without_person
            # )
    except Exception as e:
        print(e)
