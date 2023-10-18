import cv2
from tqdm import tqdm
import os
import shutil

from image_similarity import image_similarity

# Get the list of all files and directories
path = "./images"
images = os.listdir(path)

analized = []

pbar_images = tqdm(images, desc="Processing")

for image in pbar_images:
    try:
        if (os.path.isdir(image)) or (image in analized):
            continue

        images_to_analize = list(filter(lambda x: x != image, images))
        # images_to_analize = images

        new_path = f"{path}/image_similarity/{image.split('.')[0]}"
        os.makedirs(name=new_path, exist_ok=True)

        pbar_next_image = tqdm(images_to_analize)

        for next_image in pbar_next_image:
            pbar_next_image.set_description(f"Processing {image}")
            if os.path.isdir(next_image):
                continue

            if (next_image in analized):
                continue

            img1 = cv2.imread(f"{path}/{image}", 0)
            img2 = cv2.imread(f"{path}/{next_image}", 0)

            try:
                orb_similarity = image_similarity(img1, img2)
                if (orb_similarity > 0.7):
                    analized.append(next_image)
                    shutil.copy(
                        f"{path}/{next_image}",
                        new_path
                    )
            except:  # noqa: E722
                analized.append(next_image)

        analized.append(image)
        shutil.copy(
            f"{path}/{image}",
            new_path
        )

    except:  # noqa: E722
        analized.append(image)
