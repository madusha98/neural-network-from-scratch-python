import numpy as np


def sigmoid(X):
    return 1/(1+np.exp(-X))

def dsigmoid(X):
    Y = X
    for i in range(len(X)):
        for j in range(len(X[i])):
            Y[i][j] = X[i][j] * (1 - X[i][j])
    return Y

class NerualNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_ih = np.matrix(np.random.uniform(-1, 1, (self.hidden_nodes, self.input_nodes)))
        self.weights_ho = np.matrix(np.random.uniform(-1, 1, (self.output_nodes, self.hidden_nodes)))

        self.bias_h = np.matrix(np.random.uniform(-1, 1, (self.hidden_nodes, 1)))
        self.bias_o = np.matrix(np.random.uniform(-1, 1, (self.output_nodes, 1)))

        self.learning_rate = learning_rate


    def feedforward(self, input_array):
        # Generating hidden outputs
        inputs = np.asmatrix(input_array).transpose()
        hidden = self.weights_ih.dot(inputs)
        hidden += self.bias_h

        # Activation Function
        hidden = sigmoid(hidden)

        # Generating output's output
        outputs =  self.weights_ho.dot(hidden)
        outputs += self.bias_o
        outputs = sigmoid(outputs)

        return outputs

    def train(self, input_array, target_array):
        # Generating hidden outputs
        inputs = np.asmatrix(input_array).transpose()
        hidden = self.weights_ih.dot(inputs)
        hidden += self.bias_h

        # Activation Function
        hidden = sigmoid(hidden)

        # Generating output's output
        outputs =  self.weights_ho.dot(hidden)
        outputs += self.bias_o
        outputs = sigmoid(outputs)

        # convert array to matrix
        targets = np.asmatrix(target_array).transpose()

        # calculate the error
        output_errors = targets - outputs

        # calculate gradient
        gradients = dsigmoid(outputs)
        gradients = np.multiply(gradients, output_errors)
        gradients = np.multiply(gradients, self.learning_rate)

        # Calculate deltas
        hidden_T = hidden.transpose()
        weight_ho_deltas = gradients.dot(hidden_T)

        # Adjust the weights by deltas
        self.weights_ho += weight_ho_deltas
        # Adjust the bias by its deltas (which is just the gradients)
        self.bias_o += gradients

        # Calculate the hidden layer errors
        who_t = self.weights_ho.transpose()
        hidden_errors = who_t.dot(output_errors)

        # Calculate hidden gradient
        hidden_gradient = dsigmoid(hidden)
        hidden_gradient = np.multiply(hidden_gradient, hidden_errors)
        hidden_gradient = np.multiply(hidden_gradient, self.learning_rate)

        # Calcuate input->hidden deltas
        inputs_T = inputs.transpose()
        weight_ih_deltas = hidden_gradient.dot(inputs_T)

        self.weights_ih += weight_ih_deltas
        # Adjust the bias by its deltas (which is just the gradients)
        self.bias_h += hidden_gradient




        
