"""Code for generating 3 images from text prompt based on Stable Diffusion model."""
import os
import sys
import keras_cv
import matplotlib.pyplot as plt

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
if len(sys.argv) != 3:
    print('Bad usage!\n\
            python3 stable_diffusion.py <"prompt"> <path/to/output/dir/')
    sys.exit(1)
PROMPT = sys.argv[1]
RESULT_DIRECTORY = sys.argv[2]
images = model.text_to_image(PROMPT, batch_size=3)

def plot_images(imgs):
    """Show and save images."""
    plt.figure(figsize=(20, 20))
    for i, img in enumerate(imgs):
        axes = plt.subplot(1, len(imgs), i + 1)
        axes.imshow(img)
        axes.axis("off")
        plt.imsave(os.path.join(RESULT_DIRECTORY, f'image{i}.png'), img)

plot_images(images)
plt.show()
