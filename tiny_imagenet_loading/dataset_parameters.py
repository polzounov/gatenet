import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.interactive(False)


class DataSet:

    def __init__(self, path, images_in_memory):
        self.path = path
        self.image_pointers = None
        self.classes = None
        self.images_per_class = None
        self.image_size = None
        self.images = None
        self.images_in_memory = images_in_memory

        self.get_image_file_pointers()


    def get_image_file_pointers(self):
        pass

    def aquire_image(self, class_num, image_num):
        pass

    def load_images_in_memory(self):
        # if load_images_in_memory is true, read all images into a giant matrix
        # the size will be about 1.2 GB for tiny-imagenet
        self.images_in_memory = True
        self.images = np.zeros((self.classes, self.images_per_class, self.image_size), dtype=np.uint8)

        print('loading images in memory')
        for c in range(self.classes):
            for i in range(self.images_per_class):
                img = Image.open(self.image_pointers[c][i])
                img.load()
                data = np.asarray(img, dtype="uint8")

                # imgplot = plt.imshow(data)
                # plt.show()

                data = np.reshape(data, (-1,))

                # if image is grayscale then tile image to form equivalent RGB representation
                if data.shape[0] == 64 * 64:
                    data = np.tile(data, 3)

                self.images[c][i] = data

        print('images loaded')



class TinyImagenet(DataSet):
    def __init__(self, path, images_in_memory):
        super().__init__(path, images_in_memory)


    def get_image_file_pointers(self):
        # store file location to all images in image_pointers
        self.image_pointers = []

        class_folder = self.path + "/train"
        class_contents = os.listdir(class_folder)


        for i in range(len(class_contents)):
            folder = self.path + "/train/" + class_contents[i].strip() + '/images'
            contents = os.listdir(folder)
            contents = [folder + '/' + s for s in contents]
            self.image_pointers.append(contents)

        # number of classes, 200 for tiny-imagenet
        self.classes = len(self.image_pointers)

        # number of images per classes, 500 for tiny-imagenet
        self.images_per_class = len(self.image_pointers[0])

        # size of image
        self.image_size = 64 * 64 * 3

        if self.images_in_memory == True:
            self.load_images_in_memory()

    def aquire_image(self, class_num, image_num):
        if self.images_in_memory == True:
            return self.images[class_num][image_num]

        img = Image.open(self.image_pointers[class_num][image_num])
        img.load()
        data = np.asarray(img, dtype="uint8")
        data = np.reshape(data, (-1,))

        # if image is grayscale then tile image to form equivalent RGB representation
        if data.shape[0] == 64 * 64:
            data = np.tile(data, 3)

        return data





class Omniglot(DataSet):
    def __init__(self, path, images_in_memory):
        super().__init__(path, images_in_memory)


    def get_image_file_pointers(self):
        # store file location to all images in image_pointers
        self.image_pointers = []

        class_folder = self.path
        class_contents = os.listdir(class_folder)


        for i in range(len(class_contents)):

            character_folder = self.path + '/' + class_contents[i].strip()
            character_contents = os.listdir(character_folder)

            for j in range(len(character_contents)):
                folder = self.path + '/' + class_contents[i].strip() + '/' + character_contents[j].strip() + '/'
                contents = os.listdir(folder)
                contents = [folder + '/' + s for s in contents]
                self.image_pointers.append(contents)

        # number of classes, 200 for tiny-imagenet
        self.classes = len(self.image_pointers)

        # number of images per classes, 500 for tiny-imagenet
        self.images_per_class = len(self.image_pointers[0])

        # size of image
        self.image_size = 105*105

        if self.images_in_memory == True:
            self.load_images_in_memory()

    def aquire_image(self, class_num, image_num):
        if self.images_in_memory == True:
            return self.images[class_num][image_num]

        img = Image.open(self.image_pointers[class_num][image_num])
        img.load()
        data = np.asarray(img, dtype="uint8")
        data = np.reshape(data, (-1,))

        # if image is grayscale then tile image to form equivalent RGB representation
        #if data.shape[0] == 64 * 64:
         #   data = np.tile(data, 3)

        return data










def load_dataset(path, dataset='tiny-imagenet', load_images_into_memory=False):
    if dataset == 'tiny-imagenet':
        return TinyImagenet(path, load_images_into_memory)
    if dataset == 'omniglot':
        return Omniglot(path, load_images_into_memory)






'''

## Set path to tiny-imagenet folder
path = "/home/chris/images_background"
tt = Omniglot(path, False)

img = tt.aquire_image(10,10)


plt.figure(1)
imgplot = plt.imshow(np.reshape(img, (105,105)))
plt.show()
'''