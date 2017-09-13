import os
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from dataset_loading.datasets import *

plt.interactive(False)


# general class to handle dataset operations
class DataManager:

    def __init__(self, dataset='tiny-imagenet', load_images_in_memory=False):

        if dataset == 'mnist':
            self.path = "../datasets/mnist"
        elif dataset == 'omniglot':
            self.path = "../datasets/omniglot"
        elif dataset == 'tiny-imagenet':
            self.path = "../datasets/tiny-imagenet"

        # load specific dataset class
        self.dataset = load_dataset(self.path, dataset, load_images_in_memory)

        # get information about dataset
        self.classes = self.dataset.classes
        self.images_per_class = self.dataset.images_per_class
        self.image_size = self.dataset.image_size

    def build_dataset(self):
        raise NotImplementedError()

    def get_train_batch(self, meta_type='train'):
        raise NotImplementedError()

    def get_test_batch(self, meta_type='train'):
        raise NotImplementedError()

    def _meta_train_get_train_batch(self):
        raise NotImplementedError()

    def _meta_train_get_test_batch(self):
        raise NotImplementedError()

    def _meta_val_get_train_batch(self):
        raise NotImplementedError()

    def _meta_val_get_test_batch(self):
        raise NotImplementedError()

    def _meta_test_get_train_batch(self):
        raise NotImplementedError()

    def _meta_test_get_test_batch(self):
        raise NotImplementedError()


# Specific class to handle meta-dataset operations
class MetaDataManager(DataManager):
    def __init__(self,
                 dataset='tiny-imagenet',
                 load_images_in_memory=False,
                 seed=0,
                 splits={'train':0.7, 'val':0.15, 'test':0.15}):
        super().__init__(dataset, load_images_in_memory)

        # Set random seed (optional)
        if seed is not None:
            np.random.seed(seed)

        # Shuffle the indices of the classes
        shuffled_classes = np.arange(self.classes)
        np.random.shuffle(shuffled_classes)

        train_len = int(splits['train'] * self.classes)
        val_len = int(splits['val'] * self.classes)

        # Create the splits for train, val, & test
        self.classes_tr = shuffled_classes[0:train_len]
        self.classes_val = shuffled_classes[train_len:train_len+val_len]
        self.classes_ts = shuffled_classes[train_len+val_len:]

    ##### Build the datasets ###################################################
    def _build_dataset_helper(self, _classes, num_dataset_classes=5, k_shot=5, dataset_num_test=5):
        self.k_shot = k_shot
        self.num_dataset_classes = num_dataset_classes
        self.dataset_num_test = dataset_num_test
        dataset_num_train = self.images_per_class - dataset_num_test

        # Pick classes for dataset
        self.dataset_classes = np.random.choice(_classes, num_dataset_classes, replace=False)

        # Pick images for dataset
        dataset_train_images = np.empty((num_dataset_classes, dataset_num_train), dtype=np.uint8)
        dataset_train_labels = np.empty((num_dataset_classes, dataset_num_train), dtype=np.int32)
        dataset_train_classes = np.empty((num_dataset_classes, dataset_num_train), dtype=np.int32)

        dataset_test_images = np.empty((num_dataset_classes, dataset_num_test), dtype=np.uint8)
        dataset_test_labels = np.empty((num_dataset_classes, dataset_num_test), dtype=np.int32)
        dataset_test_classes = np.empty((num_dataset_classes, dataset_num_test), dtype=np.int32)

        for i in range(num_dataset_classes):
            arr = np.arange(self.images_per_class)
            np.random.shuffle(arr)

            dataset_train_images[i] = arr[:dataset_num_train]
            dataset_train_labels[i] = np.repeat(i, dataset_num_train)
            dataset_train_classes[i] = np.repeat(self.dataset_classes[i], dataset_num_train)

            dataset_test_images[i] = arr[dataset_num_train:]
            dataset_test_labels[i] = np.repeat(i, dataset_num_test)
            dataset_test_classes[i] = np.repeat(self.dataset_classes[i], dataset_num_test)

        return (dataset_train_images, dataset_train_labels, dataset_train_classes,
                dataset_test_images, dataset_test_labels, dataset_test_classes)

    def build_dataset(self, num_dataset_classes=5, k_shot=5, dataset_num_test=5):
        n, k, d = num_dataset_classes, k_shot, dataset_num_test

        # Build meta training dataset
        self.tr_dataset_train_images, self.tr_dataset_train_labels, \
        self.tr_dataset_train_classes, self.tr_dataset_test_images, \
        self.tr_dataset_test_labels, self.tr_dataset_test_classes = \
        self._build_dataset_helper(self.classes_tr, num_dataset_classes=n, k_shot=k, dataset_num_test=d)

        if len(self.classes_val) > 0:
            # Build meta validation dataset
            self.val_dataset_train_images, self.val_dataset_train_labels, \
            self.val_dataset_train_classes, self.val_dataset_test_images, \
            self.val_dataset_test_labels, self.val_dataset_test_classes = \
            self._build_dataset_helper(self.classes_val, num_dataset_classes=n, k_shot=k, dataset_num_test=d)
        
        if len(self.classes_ts) > 0:
            # Build meta testing dataset
            self.ts_dataset_train_images, self.ts_dataset_train_labels, \
            self.ts_dataset_train_classes, self.ts_dataset_test_images, \
            self.ts_dataset_test_labels, self.ts_dataset_test_classes = \
            self._build_dataset_helper(self.classes_ts, num_dataset_classes=n, k_shot=k, dataset_num_test=d)

    ##### Get the batches ######################################################
    def _get_train_batch_helper(self, ds_images, ds_labels, ds_classes):
        batch_size = self.k_shot * self.num_dataset_classes
        batch_images = np.zeros((batch_size, self.image_size), dtype=np.uint8)
        batch_labels = np.zeros((batch_size, 1), dtype=np.int32)

        count = 0
        for i in range(self.num_dataset_classes):
            elems = np.random.choice(len(ds_images[i]), self.k_shot, replace=False)
            for j in range(self.k_shot):
                batch_labels[count] = ds_labels[i][elems[j]]
                batch_images[count] = self.dataset.aquire_image(ds_classes[i][elems[j]], elems[j])
                count = count + 1

        batch_labels = np.reshape(batch_labels,(-1,))
        return batch_images, batch_labels

    def _get_test_batch_helper(self, ds_images, ds_labels, ds_classes):
        batch_size = self.dataset_num_test * self.num_dataset_classes
        batch_images = np.zeros((batch_size, self.image_size), dtype=np.uint8)
        batch_labels = np.zeros((batch_size, 1), dtype=np.int32)

        count = 0
        for i in range(self.num_dataset_classes):
            for j in range(self.dataset_num_test):
                batch_labels[count] = ds_labels[i][j]
                batch_images[count] = self.dataset.aquire_image(ds_classes[i][j], j)
                count = count + 1

        batch_labels = np.reshape(batch_labels, (-1,))
        return batch_images, batch_labels

    def _meta_train_get_train_batch(self):
        return self._get_train_batch_helper(self.tr_dataset_train_images,
                                            self.tr_dataset_train_labels,
                                            self.tr_dataset_train_classes)

    def _meta_train_get_test_batch(self):
        return self._get_test_batch_helper(self.tr_dataset_test_images,
                                           self.tr_dataset_test_labels,
                                           self.tr_dataset_test_classes)

    def _meta_val_get_train_batch(self):
        return self._get_train_batch_helper(self.val_dataset_train_images,
                                            self.val_dataset_train_labels,
                                            self.val_dataset_train_classes)

    def _meta_val_get_test_batch(self):
        return self._get_test_batch_helper(self.val_dataset_test_images,
                                           self.val_dataset_test_labels,
                                           self.val_dataset_test_classes)

    def _meta_test_get_train_batch(self):
        return self._get_train_batch_helper(self.ts_dataset_train_images,
                                            self.ts_dataset_train_labels,
                                            self.ts_dataset_train_classes)

    def _meta_test_get_test_batch(self):
        return self._get_test_batch_helper(self.ts_dataset_test_images,
                                           self.ts_dataset_test_labels,
                                           self.ts_dataset_test_classes)

    def get_train_batch(self, meta_type='train'):
        if meta_type == 'train':
            return self._meta_train_get_train_batch()
        elif meta_type == 'val':
            return self._meta_val_get_train_batch()
        elif meta_type == 'test':
            return self._meta_test_get_train_batch()
        else:
            raise ValueError('Cannot parse {}, options are: {}'.format(
                meta_type, ['train', 'val', 'test']))

    def get_test_batch(self, meta_type='train'):
        if meta_type == 'train':
            return self._meta_train_get_test_batch()
        elif meta_type == 'val':
            return self._meta_val_get_test_batch()
        elif meta_type == 'test':
            return self._meta_test_get_test_batch()
        else:
            raise ValueError('Cannot parse: {}. Options are: {}'.format(
                meta_type, ['train', 'val', 'test']))


if __name__ == '__main__':

    k_fold = 5
    num_classes = 5

    omniglot_metaDataManager = MetaDataManager(dataset='omniglot')
    omniglot_metaDataManager.build_dataset(k_fold,num_classes,5)
    images, labels = omniglot_metaDataManager.get_train_batch()
    test_images, test_labels = omniglot_metaDataManager.get_test_batch()

    plt.figure(2)
    count = 0
    for i in range(num_classes):
        for j in range(k_fold):
            plt.subplot(num_classes,k_fold,count+1)
            imgplot = plt.imshow(np.reshape(images[count], (105,105)))
            count = count + 1
    plt.show()

