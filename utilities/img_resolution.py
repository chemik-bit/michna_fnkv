from PIL import Image
import numpy as np

# Load the new image to get its dimensions
image_path_new = '/Users/honzamichna/Desktop/fig_2.png'
with Image.open(image_path_new) as img_new:
    width_new, height_new = img_new.size
    img_new = np.array(img_new)
    print(img_new)


print(width_new, height_new)
