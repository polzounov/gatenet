from dataset_loading.data_managers import *

# dataset used. Current choices: 'mnist', 'tiny-imagenet', 'omniglot'
dataset = 'mnist'
# number of classes used for training and testing
num_classes = 3
# number of images used per class for training
k_shot = 5
# number of images used per class for testing
num_testing_images = 10

# instantiate MetaDataManager class
metaDataManager = MetaDataManager(dataset=dataset,
                                  load_images_in_memory=False,
                                  seed=0,
                                  splits={'train':0.6, 'val':0., 'test':0.4})
# build new meta dataset
metaDataManager.build_dataset(num_classes, k_shot, num_testing_images)

# getting output image shape for display
if dataset == 'mnist':
    img_shape = (28, 28)
elif dataset == 'omniglot':
    img_shape = (105, 105)
elif dataset == 'tiny-imagenet':
    img_shape = (64, 64, 3)


### META TRAIN
# meta train: get training batch
meta_tr_train_images, meta_tr_train_labels = metaDataManager.get_train_batch('train')
# meta train: get test batch
meta_tr_test_images, meta_tr_test_labels = metaDataManager.get_test_batch('train')

# plot training images
fig1 = plt.figure(1)
fig1.suptitle('Meta-train: Training Images', fontsize=20)
subplot_index = 0
for i in range(num_classes):
    for j in range(k_shot):
        plt.subplot(num_classes, k_shot, subplot_index+1)
        imgplot = plt.imshow(np.reshape(meta_tr_train_images[subplot_index], img_shape))
        plt.axis('off')
        subplot_index = subplot_index + 1

# plot test images
fig2 = plt.figure(2)
fig2.suptitle('Meta-train: Test Images', fontsize=20)
subplot_index = 0
for i in range(num_classes):
    for j in range(num_testing_images):
        plt.subplot(num_classes, num_testing_images, subplot_index+1)
        imgplot = plt.imshow(np.reshape(meta_tr_test_images[subplot_index], img_shape))
        plt.axis('off')
        subplot_index = subplot_index + 1

### META TEST
# meta test: get training batch
meta_ts_train_images, meta_ts_train_labels = metaDataManager.get_train_batch('test')
# meta test: get test batch
meta_ts_test_images, meta_ts_test_labels = metaDataManager.get_test_batch('test')

# plot training images
fig7 = plt.figure(7)
fig7.suptitle('Meta-test: Training Images', fontsize=20)
subplot_index = 0
for i in range(num_classes):
    for j in range(k_shot):
        plt.subplot(num_classes, k_shot, subplot_index+1)
        imgplot = plt.imshow(np.reshape(meta_ts_train_images[subplot_index], img_shape))
        plt.axis('off')
        subplot_index = subplot_index + 1

# plot test images
fig8 = plt.figure(8)
fig8.suptitle('Meta-test: Test Images', fontsize=20)
subplot_index = 0
for i in range(num_classes):
    for j in range(num_testing_images):
        plt.subplot(num_classes, num_testing_images, subplot_index+1)
        imgplot = plt.imshow(np.reshape(meta_ts_test_images[subplot_index], img_shape))
        plt.axis('off')
        subplot_index = subplot_index + 1


'''
Notes on usage of the MetaDataManager:
    - The meta train/val/test portion if the code was added hurrily so the API 
      may be confusing
    - Whenever you want to to get new classes for a new meta-train iteration you
      must run build dataset. What this does is shuffles the dataset and gets a
      new set of classes for each of train/val/test.
        - When the MetaDataManager is initialized the dataset is split into 
          three sections so classes from meta-test will never come up for either
          of the other meta-datasets
    
    - Bugs to avoid:
        - Make sure that the splits for tr/val/ts make sense
        - Make sure k < total # of images per class
        - Make sure num_classes < split['dataset'] * total_classes
            - Only acceptable one with the split being zero is validation
'''