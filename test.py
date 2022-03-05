from neural_network import NeuralNetwork
import numpy as np
import pickle

# Load the Model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the MNIST test dataset
with open('dataset/mnist_test.csv') as test_data:
    test_data_list = test_data.readlines()

scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    # Scaling and shifting the inputs, then converting into numpy float array
    inputs = (np.asfarray(all_values[1:]) / 255 * 0.99) + 0.01

    correct_label = int(all_values[0])

    outputs = model.test(inputs)
    label = np.argmax(outputs)

    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

# Calculating performance
scorecard_array = np.asarray(scorecard)
performance = (scorecard_array.sum() / scorecard_array.size) * 100
print(f'Performance: {performance} %')
