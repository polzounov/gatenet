from dataset_loading.data_managers import *

# dataset used. Current choices: 'mnist', 'tiny-imagenet', 'omniglot'
dataset = 'mnist'

# number of classes used for training and testing
num_classes = 5

# number of images used per class for training
k_shot = 5

# number of images used per class for testing
num_testing_images = 10


# instantiate MetaDataManager class
metaDataManager = MetaDataManager(dataset=dataset)

# build new meta dataset
metaDataManager.build_dataset(k_shot,num_classes,num_testing_images)


# get training batch
train_images, train_labels = metaDataManager.get_train_batch()

# get test batch
test_images, test_labels = metaDataManager.get_test_batch()


# getting output image shape for display
if dataset == 'mnist':
    img_shape = (28,28)
elif dataset == 'omniglot':
    img_shape = (105,105)
elif dataset == 'tiny-imagenet':
    img_shape = (64,64,3)


# plot training images
fig1 = plt.figure(1)
fig1.suptitle('Training Images', fontsize=20)
subplot_index = 0
for i in range(num_classes):
    for j in range(k_shot):
        plt.subplot(num_classes,k_shot,subplot_index+1)
        imgplot = plt.imshow(np.reshape(train_images[subplot_index], img_shape))
        plt.axis('off')
        subplot_index = subplot_index + 1

# plot test images
fig2 = plt.figure(2)
fig2.suptitle('Test Images', fontsize=20)
subplot_index = 0
for i in range(num_classes):
    for j in range(num_testing_images):
        plt.subplot(num_classes,num_testing_images,subplot_index+1)
        imgplot = plt.imshow(np.reshape(test_images[subplot_index], img_shape))
        plt.axis('off')
        subplot_index = subplot_index + 1
plt.show()


