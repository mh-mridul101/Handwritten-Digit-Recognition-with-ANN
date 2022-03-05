import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # initialize the weights between input and hidden layers
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # initialize the biases between hidden and output layers
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learning_rate

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        # Hidden Layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = scipy.special.expit(hidden_inputs)
        # Output Layers
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = scipy.special.expit(final_inputs)

        # Calculate the error
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weights between output and hidden layers
        self.who += self.lr * np.dot(output_errors * final_outputs * (1 - final_outputs), hidden_outputs.T)
        # Update the weights between hidden and input layers
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)

        return np.sum(output_errors) / output_errors.size

    def test(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        # Hidden Layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = scipy.special.expit(hidden_inputs)
        # Output Layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = scipy.special.expit(final_inputs)

        return final_outputs
