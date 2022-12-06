import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import sys

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
if len(sys.argv) != 3:
    print('Bad usage!\npython3 stable_diffusion.py <"prompt"> <path/to/output/dir/')
    sys.exit(1)
prompt = sys.argv[1]
result_directory = sys.argv[2]
images = model.text_to_image(prompt, batch_size=3)


def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.imsave(os.path.join(result_directory, f'image{i}.png'), images[i])


plot_images(images)
plt.show()
