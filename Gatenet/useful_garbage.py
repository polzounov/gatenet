
##############################
# This file contains code which could be useful in future
# but in present is cluttering
##############################


'''


      number_of_tests = 3
      test_data = mnist.test.images
      test_labels = mnist.test.labels
      gates = np.zeros((3,10,10 + number_of_tests))



      elem = np.where(np.argmax(test_labels, axis=1) == digit)
      elem = elem[0][:number_of_tests]
      for i in range(number_of_tests):
        test_image = test_data[elem[i], :]
        image1 = np.reshape(test_image, (1, 28 * 28))
        gates[:, :, 10+i] = graph.determineGates(image1, x, sess)



      tr_data, tr_label = mnist.train.next_batch(50000)

      for i in range(10):
        if i == digit:
          gates[:,:,i] = np.zeros((3,10))
          continue

        elem = np.where(tr_label[:,i] == 1)
        elem = elem[0][0]
        test_image = tr_data[elem,:]
        image1 = np.reshape(test_image, (1, 28 * 28))
        gates[:,:,i] = graph.determineGates(image1, x, sess)

      np.save('gates.txt', gates)
'''