import os
import numpy as np
from PIL import Image
from dataset_loading.create_datasets import LinearProblemGenerator


def _get_data_selecting(data, wanted_classes, wanted_elements):
    ind1=[]
    ind2=[]
    for w_class in wanted_classes:
        for w_el in wanted_elements:
            ind1.append(w_class)
            ind2.append(w_el)
    return data[ind1, ind2].reshape(len(wanted_classes), len(wanted_elements))


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
                 path=None,
                 examples_per_class=None,
                 num_classes=None,
                 input_shape=None,
                 label_shape=None):
        self.path = path

        self.train_images = None # (Pointers to) the training classes
        self.train_labels = None # The training labels
        self.val_images   = None # (Pointers to) the validation classes
        self.val_labels   = None # The validation labels
        self.test_images  = None # (Pointers to) the testing classes
        self.test_labels  = None # The testing labels

        self.examples_per_class = examples_per_class # Number of images per class
        self.input_shape = input_shape # If not given assume all are different (ie mini-imagenet)
        self.label_shape = label_shape # Shape of the output
        self.num_classes = num_classes # Total number of classes in the ENTIRE dataset


class MiniImagenet(DataSet):
    def __init__(self, path):
        num_classes = 100 # Fixed for mini-imagenet
        examples_per_class = 100 # Fixed for mini-imagenet
        super(MiniImagenet, self).__init__(path=path,
                                           examples_per_class=examples_per_class,
                                           num_classes=num_classes,
                                           input_shape=None,
                                           label_shape=(100,1))
        '''Mini imagnet has 100 classes with: 64 classes in the training set, 16
        in the validation set, and 20 in the test set. So we will label them 
        sequentially. Classes 1-64 are in training, 65-80 are in validation, and
        81-100 are in testing'''

        print('\n\nCheck that the datasets can be divided easily:')
        raise ValueError('The number of num_train_classes is c={} and int(c)={}.'.format(
            0.64*num_classes,
            int(0.64*num_classes)))

        num_train_classes = int(0.64*num_classes)
        self.train_images = []
        self.train_labels = []
        with open(path + '/train.csv') as f:
            f.readline() # Skip first line
            for label_ind in range(num_train_classes):
                current_class_image = []
                current_class_label = []
                for i in range(examples_per_class):
                    line = f.readline()
                    im_path = path + '/images/' + line.split(',')[0]
                    label = path + '/images/' + line.split(',')[0]
                    current_class_image.append(im_path)
                    current_class_label.append(label)
                self.train_images.append(current_class_image)
                self.train_labels.append(current_class_label)
        self.train_images = np.array(self.train_images)
        self.train_labels = np.array(self.train_labels)

        num_val_classes = int(0.16*num_classes)
        self.val_images = []
        self.val_labels = []
        with open(path + '/val.csv') as f:
            f.readline()
            for label_ind in range(num_val_classes):
                current_class_image = []
                current_class_label = []
                for i in range(examples_per_class):
                    line = f.readline()
                    im_path = path + '/images/' + line.split(',')[0]
                    label = path + '/images/' + line.split(',')[1]
                    current_class_image.append(im_path)
                    current_class_label.append(label)
                self.val_images.append(current_class_image)
                self.val_labels.append(current_class_label)
        self.val_images = np.array(self.val_images)
        self.val_labels = np.array(self.val_labels)

        num_test_classes = int(0.20*num_classes)
        self.test_images = []
        self.test_labels = []
        with open(path + '/test.csv') as f:
            f.readline()
            for label_ind in range(num_test_classes):
                current_class_image = []
                current_class_label = []
                for i in range(examples_per_class):
                    line = f.readline()
                    im_path = path + '/images/' + line.split(',')[0]
                    label = path + '/images/' + line.split(',')[1]
                    current_class_image.append(im)
                    current_class_label.append(label)
                self.test_images.append(current_class_image)
                self.test_labels.append(current_class_label)
        self.test_images = np.array(self.test_images)
        self.test_labels = np.array(self.test_labels)

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
    
    def get_data(self, transfer_type,
                 indices=None,
                 wanted_classes=None, wanted_elements=None):
        '''Return the data matching selected classes, using either np indexing
        or by selecting classes and elements.

        Args:
            transfer_type  : Meta-set to use.
            indices        : Numpy style indices for selecting data.
            wanted_classes : Select all matching classes.
            wanted_elements: Select all matching elements.

        Return:
            The selected data.
        '''
        if wated_type.lower() is 'train':
            selected_type = self.train_classes
        elif wated_type.lower() is 'val':
            selected_type = self.val_classes
        elif wated_type.lower() is 'test':
            selected_type = self.test_classes
        else:
            raise AttributeError('Transfer tpye \'{}\' is not allowed.'
                +' Use one of [\'train\', \'val\', \'test\']')

        if indices is not None:
            return selected_type[indices]
        elif (wanted_classes is not None) and (wanted_elements is not None):
            return _get_data_selecting(selected_type, wanted_classes, wanted_elements)
        else:
            raise ValueError('Either indices or wanted_classes & wanted_elements'
                +' need to be defined. Indices={}, wanted_classes={}, & wanted_elements={}')
    """


class SimpleLinearProblem(DataSet):
    def __init__(self,
                 path,
                 shape=10,
                 examples_per_class=1000,
                 weights_mat_shapes=(3,3), # (num_layers, num_modules)
                 data_split_meta={'train': 0.64, 'val': 0.16, 'test': 0.20},
                 num_datasets=100):

        classes_per_dataset = weights_mat_shapes[1] ** weights_mat_shapes[1]
        super(SimpleLinearProblem, self).__init__(path=path,
                                                  input_shape=shape,
                                                  label_shape=shape,
                                                  examples_per_class=examples_per_class,
                                                  num_classes=classes_per_dataset*num_datasets)

        split_train = data_split_meta['train']
        split_val = data_split_meta['val']
        split_test = data_split_meta['test']
        if split_train * num_datasets * classes_per_dataset != int(split_train * num_datasets * classes_per_dataset):
            print('\nCheck that the datasets can be divided easily:\n')
            raise ValueError('The number of num_train_classes is c={} and int(c)={}.'.format(
                    split_train * num_datasets * classes_per_dataset,
                    int(split_train * num_datasets * classes_per_dataset)))

        lpg = LinearProblemGenerator(path=path,
                                     dim=shape,
                                     examples_per_class=examples_per_class,
                                     mat_shape=weights_mat_shapes,
                                     num_datasets=num_datasets,
                                     seed=None)
        ds = lpg.get_and_possibly_create_dataset(path=path,
                                                 dim=shape,
                                                 examples_per_class=examples_per_class,
                                                 mat_shape=weights_mat_shapes,
                                                 num_datasets=num_datasets,
                                                 seed=None)

        # Get training data
        num_train_classes = int(split_train*num_datasets*classes_per_dataset)
        self.train_images = ds[:num_train_classes, :, 0, :]
        self.train_labels = ds[:num_train_classes, :, 1, :]

        # Get validation data
        num_val_classes = int(split_val*num_datasets*classes_per_dataset)
        self.val_images = ds[num_train_classes:num_val_classes, :, 0, :]
        self.val_labels = ds[num_train_classes:num_val_classes, :, 1, :]

        # Get testing data
        num_test_classes = int(split_test*num_datasets*classes_per_dataset)
        self.test_images = ds[-num_test_classes:, :, 0, :]
        self.test_labels = ds[-num_test_classes:, :, 1, :]

    def get_data(self, transfer_type,
                 indices=None,
                 wanted_classes=None, wanted_elements=None):
        '''Return the data matching selected classes, using either np indexing
        or by selecting classes and elements.

        Args:
            transfer_type  : Meta-set to use.
            indices        : Numpy style indices for selecting data.
            wanted_classes : Select all matching classes.
            wanted_elements: Select all matching elements.

        Return:
            The selected data.
        '''
        if wated_type.lower() is 'train':
            selected_type = self.train_classes
        elif wated_type.lower() is 'val':
            selected_type = self.val_classes
        elif wated_type.lower() is 'test':
            selected_type = self.test_classes
        else:
            raise AttributeError('Transfer tpye \'{}\' is not allowed.'
                +' Use one of [\'train\', \'val\', \'test\']')

        if indices is not None:
            return selected_type[indices]
        elif (wanted_classes is not None) and (wanted_elements is not None):
            return _get_data_selecting(selected_type, wanted_classes, wanted_elements)
        else:
            raise ValueError('Either indices or wanted_classes & wanted_elements'
                +' need to be defined. Indices={}, wanted_classes={}, & wanted_elements={}'.format(
                    indices, wanted_classes, wanted_elements))
