import os
import numpy as np
from PIL import Image

"""
# class to store all data and functions associated with the dataset
class DataSet:

    def __init__(self, path, images_in_memory):
        self.path = path
        self.image_pointers = None
        self.classes = None
        self.examples_per_class = None
        self.image_size = None
        self.image_size_matrix = None
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
        self.images = np.zeros((self.classes, self.examples_per_class, self.image_size), dtype=np.uint8)

        print('loading images in memory')
        for c in range(self.classes):
            for i in range(self.examples_per_class):
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


# specific dataset class for tiny-imagenet
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
        self.examples_per_class = len(self.image_pointers[0])

        # size of image
        self.image_size = 64 * 64 * 3
        self.image_size_matrix = (64,64,3)

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


# specific dataset class for omniglot
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
        self.examples_per_class = len(self.image_pointers[0])

        # size of image
        self.image_size = 105*105
        self.image_size_matrix = (105,105,1)

        if self.images_in_memory == True:
            self.load_images_in_memory()

    def aquire_image(self, class_num, image_num):
        if self.images_in_memory == True:
            return self.images[class_num][image_num]

        img = Image.open(self.image_pointers[class_num][image_num])
        img.load()
        data = np.asarray(img, dtype="uint8")
        data = np.reshape(data, (-1,))

        return data


# specific dataset class for mnist
class MNIST(DataSet):
    def __init__(self, path, images_in_memory):
        super().__init__(path, images_in_memory)


    def get_image_file_pointers(self):
        # store file location to all images in image_pointers
        self.image_pointers = []

        class_folder = self.path + "/training"
        class_contents = os.listdir(class_folder)

        use_num_images = 1000

        for i in range(len(class_contents)):
            folder = self.path + "/training/" + class_contents[i].strip()
            contents = os.listdir(folder)

            contents = [folder + '/' + contents[j] for j in range(use_num_images)]
            self.image_pointers.append(contents)

        # number of classes, 200 for tiny-imagenet
        self.classes = len(self.image_pointers)

        # number of images per classes, 500 for tiny-imagenet
        self.examples_per_class = len(self.image_pointers[0])

        # size of image
        self.image_size = 28 * 28
        self.image_size_matrix = (28,28,1)

        if self.images_in_memory == True:
            self.load_images_in_memory()

    def aquire_image(self, class_num, image_num):
        if self.images_in_memory == True:
            return self.images[class_num][image_num]

        img = Image.open(self.image_pointers[class_num][image_num])
        img.load()
        data = np.asarray(img, dtype="uint8")
        data = np.reshape(data, (-1,))

        return data


# function to return specific dataset class
def load_dataset(path, dataset='tiny-imagenet', load_images_into_memory=False):
    if dataset == 'tiny-imagenet':
        return TinyImagenet(path, load_images_into_memory)
    if dataset == 'omniglot':
        return Omniglot(path, load_images_into_memory)
    if dataset == 'mnist':
        return MNIST(path, load_images_into_memory)
"""


def one_hot(num, size):
    y = np.zeros(size,)
    y[num] = 1
    return y


class DataSet:
    def __init__(self,
                 path,
                 examples_per_class,
                 input_shape,
                 label_shape):
        self.path = path

        self.train_classes = None # Pointers to the training classes
        self.val_classes   = None # Pointers to the validation classes
        self.test_classes  = None # Pointers to the testing classes

        self.examples_per_class = None # Number of images per class
        self.input_shape = None # If not given assume all are different (ie mini-imagenet)
        self.label_shape = None # Total number of classes in the ENTIRE dataset


class MiniImagenet(DataSet):
    def __init__(self, path):
        super(MiniImagenet, self).__init__(path=path,
                                           examples_per_class=600,
                                           input_shape=None,
                                           label_shape=100)
        '''Mini imagnet has 100 classes with: 64 classes in the training set, 16
        in the validation set, and 20 in the test set. So we will label them 
        sequentially. Classes 1-64 are in training, 65-80 are in validation, and
        81-100 are in testing'''
        self.train_classes = []
        with open(path + '/train.csv') as f:
            f.readline() # Skip first line
            for label_ind in range(64):
                current_class = []
                for i in range(600):
                    line = f.readline()
                    im_path = path + '/images/' + line.split(',')[0]
                    self.current_class.append(im_path)
                self.train_classes.append(current_class)
        self.train_classes = np.array(self.train_classes)

        self.val_classes = []
        with open(path + '/val.csv') as f:
            f.readline()
            for label_ind in range(16):
                current_class = []
                for i in range(600):
                    line = f.readline()
                    im_path = path + '/images/' + line.split(',')[0]
                    self.current_class.append(im_path)
                self.val_classes.append(current_class)
        self.val_classes = np.array(self.val_classes)

        self.test_classes = []
        with open(path + '/test.csv') as f:
            f.readline()
            for label_ind in range(20):
                current_class = []
                for i in range(600):
                    line = f.readline()
                    im_path = path + '/images/' + line.split(',')[0]
                    self.current_class.append(im)
                self.test_classes.append(current_class)
        self.test_classes = np.array(self.test_classes)

    """
    def get_data(self, wated_type, wanted_classes, wanted_elements):
        '''Return the data and the scalar value in the label one hot encoding
        (for classification this is always 1, for simple problem this is a 
        float)'''
        if wated_type.lower() is 'train':
            selected_type = self.train_classes
        if wated_type.lower() is 'val':
            selected_type = self.val_classes
        if wated_type.lower() is 'test':
            selected_type = self.test_classes

        selected_classes = [selected_type[cl, el][0] for cl in wanted_classes 
                                                     for el in wanted_elements]
        labels = [1 for _ in range(len(data))]

        # Todo decode the jpeg file with tensorflow and use crop and pad to fix 
        # the shape of the input
        return data, labels
    """


class SimpleLinearProblem(DataSet):
    def __init__(self,
                 path,
                 shape=10,
                 examples_per_class=600,
                 weights_mat_shapes=(3,3,3),
                 num_datasets=100):

        super(SimpleLinearProblem, self).__init__(path=path,
                                                  input_shape=shape,
                                                  label_shape=shape,
                                                  examples_per_class=examples_per_class)
        classes_per_dataset = np.prod(weights_mat_shapes)

        # Get training data
        num_train_classes = int(0.64*num_datasets*classes_per_dataset)
        self.train_classes = np.load(path + 'train.npy')
        if self.train_classes.shape != [num_train_classes, examples_per_class]:
            raise ValueError('Loaded training classes shape does not match required shape',
                '\nLoaded shape {}'.format(self.train_classes.shape),
                '\nRequired shape {}\n\n'.format([num_train_classes, examples_per_class]))

        # Get validation data
        num_val_classes = int(0.16*num_datasets*classes_per_dataset)
        self.val_classes = np.load(path + 'val.npy')
        if self.val_classes.shape != [num_val_classes, examples_per_class]:
            raise ValueError('Loaded validation classes shape does not match required shape',
                '\nLoaded shape {}'.format(self.val_classes.shape),
                '\nRequired shape {}\n\n'.format([num_val_classes, examples_per_class]))

        # Get testing data
        num_test_classes = int(0.20*num_datasets*classes_per_dataset)
        self.test_classes = np.load(path + 'test.npy')
        if self.test_classes.shape != [num_test_classes, examples_per_class]:
            raise ValueError('Loaded testing classes shape does not match required shape',
                '\nLoaded shape {}'.format(self.test_classes.shape),
                '\nRequired shape {}\n\n'.format([num_test_classes, examples_per_class]))