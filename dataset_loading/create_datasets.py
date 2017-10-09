import numpy as np

# Honestly no reason this should be a class, change to funcs
class LinearProblemGenerator(object):
    def __init__(self,
                 path=None,
                 dim=5, # Input/Ouput/Hidden size
                 examples_per_class=1000,
                 mat_shape=(3,3),  # (num_layers, num_modules)
                 num_datasets=1, # classes = num_ds * (num_modules ^ num_layers)
                 seed=None):
        self.seed=seed
        self.dim = dim
        self.layers, self.modules = mat_shape
        self.examples_per_class = examples_per_class

        self.random_mat_func = lambda x: np.random.normal(loc=.01, size=(x, x))
        self.random_input_func = lambda n, d: np.random.normal(loc=10., scale=10., size=(n, d))

    def _create_random_weight_matrix(self):
        '''Creates (l,m) weight matrices shaped dim, optional seed'''
        #weight_mat = np.zeros((self.layers, self.modules))
        weight_mat = [[0 for _ in range(self.modules)] for _ in range(self.layers)]
        print('layers', self.layers)
        print('modules', self.modules)
        for layer in range(self.layers):
            for module in range(self.modules):
                if self.seed is not None:
                    # You can create the same dataset multiple times if the 
                    # starting seed is set the same, and the order of random
                    # matrix creation is always the same
                    numpy.random.seed(self.seed)
                    self.seed += 1 # Diff matrices will be diff values
                weight_mat[layer][module] = self.random_mat_func(self.dim)
        return weight_mat

    def _create_random_input(self):
        if self.seed is not None:
            # You can create the same dataset multiple times if the 
            # starting seed is set the same, and the order of random
            # matrix creation is always the same
            numpy.random.seed(self.seed)
            self.seed += 1 # Diff matrices will be diff values
        return self.random_input_func(self.examples_per_class, self.dim)

    def _get_combinations(self):
        '''
        a  = [[(0,0), (1,0), (2,0)],
              [(0,0), (1,0), (2,1)],
              [(0,0), (1,0), (2,2)],
              [(0,0), (1,1), (2,0)],
              [(0,0), (1,1), (2,1)],
              [(0,0), (1,1), (2,2)],
              [(0,0), (1,2), (2,0)],
              [(0,0), (1,2), (2,1)],
              [(0,0), (1,2), (2,2)],

              [(0,1), (1,0), (2,0)],
              [(0,1), (1,0), (2,1)],
              [(0,1), (1,0), (2,2)],
              [(0,1), (1,1), (2,0)],
              [(0,1), (1,1), (2,1)],
              [(0,1), (1,1), (2,2)],
              [(0,1), (1,2), (2,0)],
              [(0,1), (1,2), (2,1)],
              [(0,1), (1,2), (2,2)],

              [(0,2), (1,0), (2,0)],
              [(0,2), (1,0), (2,1)],
              [(0,2), (1,0), (2,2)],
              [(0,2), (1,1), (2,0)],
              [(0,2), (1,1), (2,1)],
              [(0,2), (1,1), (2,2)],
              [(0,2), (1,2), (2,0)],
              [(0,2), (1,2), (2,1)],
              [(0,2), (1,2), (2,2)]]
        '''
        a  = [[(0,0), (1,0), (2,0)]]
        #'''
        return a

    def _create_single_dataset(self):
        # cl, el, dim
        weight_mat = np.array(self._create_random_weight_matrix())
        combimations = self._get_combinations()

        x = np.zeros((self.modules**self.layers, self.examples_per_class, self.dim))
        y = np.zeros((self.modules**self.layers, self.examples_per_class, self.dim))
        for i, comb in enumerate(combimations):
            # Get a single matrix representing the whole function to save computation
            print(comb)
            print(weight_mat.shape)
            weights = [weight_mat[ind] for ind in comb]
            combined_weight = weights[0]
            for w in weights[1:]:
                combined_weight = combined_weight.dot(w)

            # X & Y are flipped because we want to learn the inverse tranforms
            y[i] = self._create_random_input() # (examples_per_class by dim)
            print(y[i].shape)
            print(x[i].shape)
            x[i] = y[i].dot(combined_weight) # should be (examples_per_class by dim)
        return x, y

    def get_datasets(self, n=1, combine=False):
        '''
        Args:
            n: number of datasets to create 
               (classes = n * modules ^ layers)
            combine: Return X & Y as two seperate matrices or as one (default False)
        '''
        num_classes = n * (self.modules ** self.layers)

        if combine:
            ds = np.zeros((num_classes, self.examples_per_class, 2, self.dim))
            for i in range(n):
                ind_1 = i * (self.modules ** self.layers)
                ind_2 = (i+1) * (self.modules ** self.layers)
                ds[ind_1:ind_2,:,0,:], ds[ind_1:ind_2,:,1,:] = self._create_single_dataset()
            return ds
        else:
            x = np.zeros((num_classes, self.examples_per_class, self.dim))
            y = np.zeros((num_classes, self.examples_per_class, self.dim))
            for i in range(n):
                ind_1 = i * (self.modules ** self.layers)
                ind_2 = (i+1) * (self.modules ** self.layers)
                x[ind_1:ind_2], y[ind_1:ind_2] = self._create_single_dataset()
            return x, y