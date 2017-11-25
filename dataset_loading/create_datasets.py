import numpy as np
import os
from hashlib import md5

def _generate_random_weights_matrix(n_modules, n_layers, dims, random_func, seed):
    '''Generates an matrix of numpy ndarrays. Where each ndarray has shape 
    (dims).

    Args:
        n_modules: Number of matrices per column.
        n_layers : Number of matrices per row.
        dims     : Shape of the ndarrays.
        rand_func: Takes a tuple and generates a random ndarray with that 
                   shape. 
        seed     : Optional seed, so the generation of the matrix is 
                   deterministic
    Return:
        The matrix of random ndarrays.
    '''
    weight_mat = [[0 for _ in range(n_modules)] for _ in range(n_layers)]
    for layer in range(n_layers):
        for module in range(n_modules):
            if seed is not None:
                np.random.seed(seed)
                seed += 1
            weight_mat[layer][module] = random_func(dims)
    return weight_mat


def _generate_random_input(random_func, shape, seed):
    '''Generates an matrix of numpy ndarrays. Where each ndarray has shape 
    (dims).

    Args:
        shape    : Shape of the input
        rand_func: Takes a tuple and generates a random ndarray with that 
                   shape. 
        seed     : Optional seed, so the generation of the matrix is 
                   deterministic
    Return:
        The random matrix.
    '''
    if seed is not None:
        np.random.seed(seed)
        seed += 1 # Diff matrices will be diff values
    return random_func(shape)


def _get_combinations(shape):
    '''Returns possible pathways in a matrix.

    Args:
        shape: Shape of the matrix
    Return: 
        A matrix with rows representing the pathways (an element in each column
        is a tuple representing an 'stop' in the pathway)
    '''

    if shape == (3,3):
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
    elif shape == (3,1):
        a  = [[(0,0), (1,0), (2,0)]]
    else:
        raise NotImplementedError('The function _get_combinations is not yet '
            +'implemented, use either shape=(3,3) or shape=(3,1)')
    return a


class LinearProblemGenerator(object):
    def __init__(self,
                 path=None,
                 dim=5, # Input/Ouput/Hidden size
                 examples_per_class=1000,
                 mat_shape=(3,3),  # (num_layers, num_modules)
                 num_datasets=1, # classes = num_ds * (num_modules ^ num_layers)
                 seed=None):
        self.path = path
        self.dim = dim
        self.examples_per_class = examples_per_class
        self.layers, self.modules = mat_shape
        self.num_datasets = num_datasets
        self.seed=seed

        self.random_mat_func = lambda shape: np.random.normal(loc=.01, size=shape)
        self.random_input_func = lambda shape: np.random.normal(loc=10., scale=10., size=shape)

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
                    np.random.seed(self.seed)
                    self.seed += 1 # Diff matrices will be diff values
                weight_mat[layer][module] = self.random_mat_func(self.dim)
        return weight_mat

    def _create_random_input(self):
        if self.seed is not None:
            # You can create the same dataset multiple times if the 
            # starting seed is set the same, and the order of random
            # matrix creation is always the same
            np.random.seed(self.seed)
            self.seed += 1 # Diff matrices will be diff values
        return self.random_input_func(self.examples_per_class, self.dim)

    def _create_single_dataset(self):
        # cl, el, dim
        weight_mat = np.array(
            _generate_random_weights_matrix(self.modules,
                                            self.layers, 
                                            (self.dim, self.dim),
                                            self.random_mat_func,
                                            self.seed))
        combimations = _get_combinations((self.layers, self.modules))

        x = np.zeros((self.modules**self.layers, self.examples_per_class, self.dim))
        y = np.zeros((self.modules**self.layers, self.examples_per_class, self.dim))
        for i, comb in enumerate(combimations):
            # Get a single matrix representing the (n_layers) matrix multiplies to save computation
            weights = [weight_mat[ind] for ind in comb]
            combined_weight = weights[0]
            for w in weights[1:]:
                combined_weight = combined_weight.dot(w)

            # X & Y are flipped because we want to learn the inverse tranforms
            y[i] = _generate_random_input(self.random_input_func,
                                          (self.examples_per_class,self.dim),
                                          self.seed)
            x[i] = y[i].dot(combined_weight)
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

    def get_and_possibly_create_dataset(self,
                                        path=None,
                                        dim=None, # Input/Ouput/Hidden size
                                        examples_per_class=None,
                                        mat_shape=None, # (num_layers, num_modules)
                                        num_datasets=None, # classes = num_ds * (num_modules ^ num_layers)
                                        seed=None):
        '''Return the dataset that is saved in path. If no dataset exists,
        create one and save it to path.

        Args:
            path              : Path (directory) of the dataset
            dim               : Input/Ouput/Hidden size
            examples_per_class: Number of examples to create per class
            mat_shape         : (num_layers, num_modules)
            num_datasets      : Number of seperate datasets to create.
            seed              : Optional seed.

        Returns:
            ds: A 3d-array (class, example, dim_of in/output)
                Will be shaped: (total num classes, examples per class, dim)
        '''
        # If any of args to this func are not None update the class variables
        self.path = path or self.path
        self.dim  = dim  or self.dim
        self.examples_per_class = examples_per_class or self.examples_per_class
        self.layers, self.modules = mat_shape or (self.layers, self.modules)
        self.num_datasets = num_datasets or self.num_datasets
        self.seed = seed or self.seed

        # Makes sure that different datasets have different names (don't use
        # an old dataset by accident)
        var_str  = (str(dim) + str(examples_per_class) + str(mat_shape) + str(num_datasets)).encode('utf-8')
        var_hash = md5(var_str).hexdigest()
        filename = path + var_hash + '.npy'

        if os.path.isfile(filename): # If file already exists
            return np.load(filename)
        else: # Create new dataset and save it
            print('\nCreating new dataset, with the following parameters:',
                  '\n\tDim is               : {}'.format(self.dim),
                  '\n\tExamples per class is: {}'.format(self.examples_per_class),
                  '\n\tMat shape is         : {}'.format((self.layers, self.modules)),
                  '\n\tNum dataset is       : {}'.format(self.num_datasets),
                  '\n\nThe hash is: {}'.format(var_hash),
                  '\n\n'
                  )
            if seed is not None:
                self.seed = seed
            num_layers, num_modules = mat_shape
            num_classes = (num_modules ** num_layers) * num_datasets

            ds = self.get_datasets(n=num_datasets, combine=True)
            np.save(filename, ds)
            return ds
