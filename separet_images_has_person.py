import os

from tinydb import Query, TinyDB
from tqdm import tqdm

from person_detection import person_detection

path = "./images"

database = TinyDB(f"{path}/database.json", indent=4)
Busca = Query()

images = os.listdir(path)

print(len(images))

pbar_images = tqdm(images, desc="Processing")

for image in pbar_images:
    try:
        if os.path.isdir(f"{path}/{image}"):
            continue

        alreadyAnalized = database.search(Busca.image == image)

        if len(alreadyAnalized) > 0:
            continue

        people = person_detection(f"{path}/{image}")

        if (people > 0):
            database.insert({
                "image": image,
                "hasPerson": True
            })
        else:
            database.insert({
                "image": image,
                "hasPerson": False
            })
    except Exception as e:
        print(e)
