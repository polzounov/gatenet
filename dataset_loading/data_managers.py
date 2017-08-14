import os
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from dataset_loading.datasets import *

plt.interactive(False)


# general class to handle dataset operations
class DataManager:

    def __init__(self, path, dataset='tiny-imagenet', load_images_in_memory = False ):

        # path to dataset folder
        self.path = path

        # load specific dataset class
        self.dataset = load_dataset(path, dataset, load_images_in_memory)

        # get information about dataset
        self.classes = self.dataset.classes
        self.images_per_class = self.dataset.images_per_class
        self.image_size = self.dataset.image_size

    def build_dataset(self):
        pass

    def get_train_batch(self):
        pass



# specific class to handle meta-dataset operations
class MetaDataManager(DataManager):
    def __init__(self,path, dataset='tiny-imagenet', load_images_in_memory = False):
        super().__init__(path, dataset, load_images_in_memory)


    def build_dataset(self, num_dataset_classes=5, k_shot=5, dataset_num_test=5 ):
        self.k_shot = k_shot
        self.num_dataset_classes = num_dataset_classes
        self.dataset_num_test = dataset_num_test

        # pick classes for dataset
        self.dataset_classes = np.random.choice(self.classes, self.num_dataset_classes, replace=False)

        # pick images for dataset
        self.dataset_train_images = np.empty((self.num_dataset_classes, self.images_per_class - dataset_num_test),
                                             dtype=np.uint8)
        self.dataset_train_labels = np.empty((self.num_dataset_classes, self.images_per_class - dataset_num_test),
                                             dtype=np.int32)
        self.dataset_train_classes = np.empty((self.num_dataset_classes, self.images_per_class - dataset_num_test),
                                             dtype=np.int32)

        self.dataset_test_images = np.empty((self.num_dataset_classes, dataset_num_test), dtype=np.uint8)
        self.dataset_test_labels = np.empty((self.num_dataset_classes, dataset_num_test), dtype=np.int32)

        for i in range(self.num_dataset_classes):
            arr = np.arange(self.images_per_class)
            np.random.shuffle(arr)

            self.dataset_train_images[i] = arr[:self.images_per_class - dataset_num_test]
            self.dataset_train_labels[i] = np.repeat(i, self.images_per_class - dataset_num_test)
            self.dataset_train_classes[i] = np.repeat(self.dataset_classes[i], self.images_per_class - dataset_num_test)

            self.dataset_test_images[i] = arr[self.images_per_class - dataset_num_test:]
            self.dataset_test_labels[i] = np.repeat(i, dataset_num_test)

    def get_train_batch(self):
        batch_size = self.k_shot * self.num_dataset_classes
        batch_images = np.zeros((batch_size, self.image_size), dtype=np.uint8)
        batch_labels = np.zeros((batch_size, 1), dtype=np.int32)

        count = 0
        for i in range(self.num_dataset_classes):
            elems = np.random.choice(len(self.dataset_train_images[i]), self.k_shot, replace=False)
            for j in range(self.k_shot):
                batch_labels[count] = self.dataset_train_labels[i][elems[j]]
                batch_images[count] = self.dataset.aquire_image(self.dataset_train_classes[i][elems[j]], elems[j])
                count = count + 1

        batch_labels = np.reshape(batch_labels,(-1,))
        return batch_images, batch_labels



if __name__ == '__main__':

    k_fold = 5
    num_classes = 5

    tiny_imagenet_path = "/home/chris/tiny-imagenet-200"
    omniglot_path = "/home/chris/images_background"

    imagenet_metaDataManager = MetaDataManager(tiny_imagenet_path, dataset='tiny-imagenet')
    imagenet_metaDataManager.build_dataset(k_fold,num_classes,100)
    images, labels = imagenet_metaDataManager.get_train_batch()

    plt.figure(1)
    count = 0
    for i in range(num_classes):
        for j in range(k_fold):
            plt.subplot(num_classes,k_fold,count+1)
            imgplot = plt.imshow(np.reshape(images[count], (64,64,3)))
            count = count + 1
    #plt.show()

    omniglot_metaDataManager = MetaDataManager(omniglot_path,dataset='omniglot')
    omniglot_metaDataManager.build_dataset(k_fold,num_classes,5)
    images, labels = omniglot_metaDataManager.get_train_batch()


    plt.figure(2)
    count = 0
    for i in range(num_classes):
        for j in range(k_fold):
            plt.subplot(num_classes,k_fold,count+1)
            imgplot = plt.imshow(np.reshape(images[count], (105,105)))
            count = count + 1
    plt.show()

