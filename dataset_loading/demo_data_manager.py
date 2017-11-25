import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from dataset_loading.data_managers import SimpleProbTransferLearnDataManager

'''
def demo_transfer_miniimagenet():
    shape = (100,100) # This is not a fixed size so this need to be selected in datasets 
    task_classes = 2
    batch_size = 5
    sp = MiniImagenetTransferLearnDataManager(dataset='simple-linear',
                                              num_transfer_classes=10,
                                              task_classes=task_classes,
                                              data_split_meta={'train': 0.6, 'val': 0.0, 'test': 0.4})

    sp.build_dataset('train')
    sp.build_task()
    x, y = sp.get_train_batch(batch_size=batch_size)

    # plot test images
    fig1 = plt.figure(1)
    fig1.suptitle('Meta-train: Train Inputs', fontsize=20)
    subplot_index = 0
    for i in range(task_classes):
        for j in range(batch_size):
            plt.subplot(task_classes, batch_size, subplot_index+1)
            imgplot = plt.imshow(np.reshape(x[subplot_index,:], shape))
            plt.axis('off')
            subplot_index = subplot_index + 1

    fig2 = plt.figure(2)
    fig2.suptitle('Meta-train: Train Labels', fontsize=20)
    subplot_index = 0
    for i in range(task_classes):
        for j in range(batch_size):
            plt.subplot(task_classes, batch_size, subplot_index+1)
            imgplot = plt.imshow(np.reshape(y[subplot_index,:], shape))
            plt.axis('off')
            subplot_index = subplot_index + 1
    plt.show()
'''


def demo_transfer_simpleprob():
    shape = (2,5)
    task_classes = 1
    batch_size = 5
    sp = SimpleProbTransferLearnDataManager(dataset='simple-linear',
                                            num_transfer_classes=27,
                                            task_classes=task_classes,
                                            data_split_meta={'train': 0.75, 'val': 0.00, 'test': 0.25},
                                            examples_per_class=100,
                                            num_datasets=4)
    sp.build_dataset('train')
    sp.build_task()
    x, y = sp.get_train_batch(batch_size=batch_size)

    # plot test images
    fig1 = plt.figure(1)
    fig1.suptitle('Meta-train: Train Inputs', fontsize=20)
    subplot_index = 0
    for i in range(task_classes):
        for j in range(batch_size):
            plt.subplot(task_classes, batch_size, subplot_index+1)
            imgplot = plt.imshow(np.reshape(x[subplot_index,:], shape))
            plt.axis('off')
            subplot_index = subplot_index + 1

    fig2 = plt.figure(2)
    fig2.suptitle('Meta-train: Train Labels', fontsize=20)
    subplot_index = 0
    for i in range(task_classes):
        for j in range(batch_size):
            plt.subplot(task_classes, batch_size, subplot_index+1)
            imgplot = plt.imshow(np.reshape(y[subplot_index,:], shape))
            plt.axis('off')
            subplot_index = subplot_index + 1
    plt.show()


if __name__ == '__main__':
    #demo_transfer_miniimagenet()
    demo_transfer_simpleprob()