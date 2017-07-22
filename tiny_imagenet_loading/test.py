import os
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt

plt.interactive(False)

#########################
#########################
#########################

## Set path to tiny-imagenet folder
path = "/home/chris/tiny-imagenet-200"

#########################
#########################


text_file = open(path + "/wnids.txt", "r")
lines = text_file.readlines()

image_pointers = []
for i in range(len(lines)):
    folder = path + "/train/" + lines[i].strip() + '/images'
    contents = os.listdir(folder)
    contents = [folder + '/' + s for s in contents]
    image_pointers.append(contents)


classes = len(image_pointers)
images_per_class = len(image_pointers[0])
image_size = 64*64*3



start = time.time()

images = np.zeros((classes, images_per_class, image_size), dtype=np.uint8)


for c in range(classes):
    print(c)
    for i in range(images_per_class):
        img = Image.open(image_pointers[c][i])
        img.load()
        data = np.asarray(img, dtype="uint8")

        #imgplot = plt.imshow(data)
        #plt.show()

        data = np.reshape(data, (-1,))
        if data.shape[0] == 64*64:
            data = np.tile(data, 3)

        images[c][i] = data



end = time.time()
print(end - start)










