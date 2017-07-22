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


class DataManager:

    def __init__(self, path, load_images_in_memory = False ):

        # set path to tiny-imagenet folder
        self.path = path


        # load in class folder names
        text_file = open(path + "/wnids.txt", "r")
        lines = text_file.readlines()

        # store file location to all images in image_pointers
        self.image_pointers = []
        for i in range(len(lines)):
            folder = self.path + "/train/" + lines[i].strip() + '/images'
            contents = os.listdir(folder)
            contents = [folder + '/' + s for s in contents]
            self.image_pointers.append(contents)


        # number of classes, 200 for tiny-imagenet
        self.classes = len(self.image_pointers)

        # number of images per classes, 500 for tiny-imagenet
        self.images_per_class = len(self.image_pointers[0])

        # size of image
        self.image_size = 64 * 64 * 3

        # if load_images_in_memory is true, read all images into a giant matrix
        # the size will be about 1.2 GB for tiny-imagenet
        self.load_images_in_memory = load_images_in_memory
        if self.load_images_in_memory == True:
            self.images = np.zeros((self.classes, self.images_per_class, self.image_size), dtype=np.uint8)

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
        else:
            self.images = None


    def build_dataset(self, num_dataset_classes=5, k_shot=5, dataset_num_test=100 ):
        self.k_shot = k_shot
        self.num_dataset_classes = num_dataset_classes
        self.dataset_num_test = dataset_num_test


        # pick classes for dataset
        self.dataset_classes = np.random.choice(self.classes, self.num_dataset_classes , replace=False)


        # pick images for dataset
        self.dataset_train_images = np.empty((self.num_dataset_classes, self.images_per_class-dataset_num_test), dtype=np.uint8)
        self.dataset_train_labels = np.empty((self.num_dataset_classes, self.images_per_class-dataset_num_test), dtype=np.int32)

        self.dataset_test_images = np.empty((self.num_dataset_classes, dataset_num_test), dtype=np.uint8)
        self.dataset_test_labels = np.empty((self.num_dataset_classes, dataset_num_test), dtype=np.int32)

        for i in range(self.num_dataset_classes):
            arr = np.arange(self.images_per_class)
            np.random.shuffle(arr)

            self.dataset_train_images[i] = arr[:self.images_per_class-dataset_num_test]
            self.dataset_train_labels[i] = np.repeat(self.dataset_classes[i], self.images_per_class-dataset_num_test)

            self.dataset_test_images[i] = arr[self.images_per_class-dataset_num_test:]
            self.dataset_test_labels[i] = np.repeat(self.dataset_classes[i], dataset_num_test)


    def get_train_batch(self):

        batch_size = self.k_shot*self.num_dataset_classes
        batch_images = np.zeros((batch_size, self.image_size), dtype=np.uint8)
        batch_labels = np.zeros((batch_size,1), dtype=np.int32)

        count = 0
        for i in range(self.num_dataset_classes):
            elems = np.random.choice(len(self.dataset_train_images[i]), self.k_shot , replace=False)
            for j in range(self.k_shot):
                batch_labels[count] = self.dataset_train_labels[i][elems[j]]
                batch_images[count] = self.aquire_images(self.dataset_train_labels[i][elems[j]], elems[j])
                count = count+1

        return batch_images, batch_labels


    def aquire_images(self, image_class, image_number):
        img = Image.open(self.image_pointers[image_class][image_number])
        img.load()
        data = np.asarray(img, dtype="uint8")

        # imgplot = plt.imshow(data)
        # plt.show()

        data = np.reshape(data, (-1,))

        # if image is grayscale then tile image to form equivalent RGB representation
        if data.shape[0] == 64 * 64:
            data = np.tile(data, 3)

        return data






k_fold = 5
num_classes = 10
dataManager = DataManager(path)
dataManager.build_dataset(k_fold,num_classes,100)
images, labels = dataManager.get_train_batch()


plt.figure(1)
count = 0
for i in range(num_classes):
    for j in range(k_fold):
        plt.subplot(num_classes,k_fold,count+1)
        imgplot = plt.imshow(np.reshape(images[count], (64,64,3)))
        count = count + 1
plt.show()



gg = 1






