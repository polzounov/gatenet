import pickle


class Parameters:
    def __init__(self):
        self.M = 10
        self.L = 3
        self.tensor_size = 20
        self.gamma = 2
        self.batch_size = 100
        self.num_batches = 10000
        self.learning_rate = 0.001
        self.output_file = 'test'


class Testing:
    def __init__(self):
        self.tests = None

    def setupTests(self):
        self.tests = []

        # Define Test 1
        test1 = Parameters()
        test1.output_file = 'test1'
        self.tests.append(test1)

        # Define Test 2
        test2 = Parameters()
        test2.gamma = 20
        test2.output_file = 'test2'
        self.tests.append(test2)


class OutputManager:

    def __init__(self):
        self.output_file = None
        self.parameter_dict = None
        self.data = None
        self.labels = None


    def initialize(self, parameter_dict):
        self.parameter_dict = parameter_dict
        self.data = []
        self.labels = []

    def addData(self, data, label):
        self.data.append(data)
        self.labels.append(label)

    def save(self):
        output = (self.data, self.labels, self.parameter_dict)
        with open('output/' + self.parameter_dict['output_file'] + '.pkl', 'wb') as f:
            pickle.dump(output, f , pickle.HIGHEST_PROTOCOL)

