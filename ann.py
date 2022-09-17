import numpy as np 

np.random.seed(1)


INPUT_SIZE = 21
TRAINING_SET_SIZE = 150
TESTING_SET_SIZE = 2000
ENABLE_BIAS = True
BATCHES = 1000
LEARNING_RATE = 0.1


class NeuralNetwork:
    def __init__(self, input_size, training_size, testing_size, enable_bias, batches, learning_rate):
        self.input_size = input_size
        self.output_size = 1
        self.training_set_size = training_size
        self.testing_set_size = testing_size
        self.enable_bias = enable_bias
        self.batches = batches
        self.learning_rate = learning_rate
        self.weights = self.init_weights()
        self.training_inputs, self.training_outputs = self.init_set(self.training_set_size)
        self.testing_inputs, self.testing_outputs = self.init_set(self.testing_set_size)

    'Initializing the weights randomly'
    def init_weights(self):
        if self.enable_bias == True:
            weights = 2 * np.random.rand(self.input_size + 1, 1) - 1
        else:
            weights = 2 * np.random.rand(self.input_size, 1) - 1
        return weights

    'Creating training and testing sets'
    def init_set(self, size):

        'Creating an input array, 21 digits of 0 and 1,  size times'
        inputs = np.array([np.random.choice([0, 1], size=self.input_size) for _ in range(size)])

        'Adding the bias to the inputs'
        if self.enable_bias:  
            inputs = np.insert(inputs, self.input_size, 1, axis=1)

        'Getting the outputs of the inputs, by adding 1 to count when the digit is 1 and -1 when its 0'
        outputs = np.array([])
        for i in inputs:
            count = 0
            for j in i:
                if j == 0:  count -= 1
                else:   count += 1
            if (count <= 0):     outputs = np.append(outputs, 0) 
            else:   outputs = np.append(outputs, 1) 
        outputs = np.array([outputs]) 
        outputs = outputs.T
    
        return inputs, outputs

    'Running the input through the network and getting the result'
    def forward(self, input_):
        return self.sigmoid(np.dot(input_, self.weights))

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x*(1-x)

    'Training the network'
    def training(self):
        for batch in range(self.batches):
            'forward propogation'
            input_layer = self.training_inputs
            outputs = self.forward(input_layer)
           
            'getting the error'
            error = self.training_outputs - outputs
            if not batch % 100:
                print("Batch: {},   while training the error: {:.4f},   while testing the error: {:.4f}".format(batch, np.mean(np.abs(error)), self.testing()))

            'calculating delta'
            delta = error * self.sigmoid_derivative(outputs)

            'updating the weights'
            self.weights += self.learning_rate * np.dot(input_layer.T, delta)
            

    def testing(self):
        input_layer = self.testing_inputs
        outputs = self.sigmoid(np.dot(input_layer, self.weights))
        error = self.testing_outputs - outputs
        return np.mean(np.abs(error))

def main():
    
    n = NeuralNetwork(INPUT_SIZE, TRAINING_SET_SIZE, TESTING_SET_SIZE, ENABLE_BIAS, BATCHES, LEARNING_RATE)

    print("Network is traning on ", TRAINING_SET_SIZE, " examples.")
    n.training()
    print("Training is done.")

    print("Network is testing ", TESTING_SET_SIZE, " examples.")
    error_rate = n.testing()
    print("The testing error is: {:.4f}".format(error_rate))

    print("PLease enter {} digits number of 1 or 0:".format(INPUT_SIZE))
    
    while True:
        s = input()

        if not s:
            return

        'Checking if the input is legal'
        if len(s) != INPUT_SIZE or not s.isdigit() or set(s) - set(("0", "1")):
            print("Input must be a {} digits long number of 0  ana 1:".format(INPUT_SIZE))
            continue

        'Convert input to numpy array'
        input_ = np.array(list(s), dtype=int)
        if ENABLE_BIAS:
            input_ = np.append(input_, 1)

        ans = n.forward(input_)[0]
        print("Answer: {} ({:.3f})".format(ans >= 0.5, ans))

if __name__ == "__main__":
    main()
