from neural_network import NeuralNetwork
import numpy as np
import pickle


input_nodes = 784
hidden_nodes = 250
output_nodes = 10
learning_rate = 0.1

epochs = 10

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Loading MNIST dataset for training
with open('dataset/mnist_train.csv') as training_data:
    training_data_list = training_data.readlines()

for epoch in range(epochs):
    count = 0
    print(f'Training ================================>epoch:{epoch + 1}')
    for record in training_data_list:
        all_values = record.split(',')
        # Scaling and shifting the inputs, then converting into numpy float array
        inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01

        # Creating targets label [0.01 - 0.99]
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99

        error = n.train(inputs, targets)

        if count % 10000 == 0:
            print(f'Error:------------> {error*100} %')
        count += 1

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(n, f)
