import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

w0 = pd.read_csv("C:/Users/REDTECH/Desktop/DNN Assignment/Assignment_1/Task_1/a/w.csv", header=None)
b0 = pd.read_csv("C:/Users/REDTECH/Desktop/DNN Assignment/Assignment_1/Task_1/a/b.csv", header=None)
w1 = pd.read_csv(r"C:\Users\REDTECH\Desktop\DNN Assignment\Assignment_1\Task_1\b\w-100-40-4.csv", header=None)
b1 = pd.read_csv(r"C:\Users\REDTECH\Desktop\DNN Assignment\Assignment_1\Task_1\b\b-100-40-4.csv", header=None)

weight_set = w1
bias_set = b1

weights_btw_layer0_to_layer1 = weight_set.iloc[:14, 1:].values  #(14, 100)
bias_for_layer1 = bias_set.iloc[:1, 1:].values  #(100,)

weights_btw_layer1_to_layer2 = weight_set.iloc[14:114, 1:41].values   #100,40
bias_for_layer2 = bias_set.iloc[1:2, 1:41].values  #(40,)

weights_btw_layer2_to_layer3 = weight_set.iloc[114:, 1:5].values   #40,4
bias_for_layer3 = bias_set.iloc[2:, 1:5].values  #(4,)

# Store initial parameters (weights and biases)
initial_params = {
    'W0': weights_btw_layer0_to_layer1,
    'b0': bias_for_layer1,
    'W1': weights_btw_layer1_to_layer2,
    'b1': bias_for_layer2,
    'W2': weights_btw_layer2_to_layer3,
    'b2': bias_for_layer3
}

# X=[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]
# X = pd.DataFrame(X)
# X_train = X.T
# y_train = [3]
# y_train = pd.DataFrame(y_train)

# Load training and test data
X_train = pd.read_csv(r"C:\Users\REDTECH\Desktop\DNN Assignment\Assignment_1\Task_2\x_train.csv", header=None)
y_train = pd.read_csv(r"C:\Users\REDTECH\Desktop\DNN Assignment\Assignment_1\Task_2\y_train.csv", header=None)
X_test = pd.read_csv(r"C:\Users\REDTECH\Desktop\DNN Assignment\Assignment_1\Task_2\x_test.csv", header=None)
y_test = pd.read_csv(r"C:\Users\REDTECH\Desktop\DNN Assignment\Assignment_1\Task_2\y_test.csv", header=None)

def save_gradients_to_csv(gradients):
    with open("dw.csv", 'w') as dW_file, open("db.csv", 'w') as db_file:
        for gradient in gradients:
            dW = gradient['dW']
            db = gradient['db']

            np.savetxt(dW_file, dW, delimiter=',', fmt='%.16e')

            np.savetxt(db_file, db, delimiter=',', fmt='%.16e')
class Network:
    def __init__(self, layers, initial_params):
        self.layers = layers
        self.params = self.initialize_params(initial_params)
        self.activations = [self.get_activation(layer['activation']) for layer in self.layers]

    def initialize_params(self, initial_params):
        params = []
        for i, layer in enumerate(self.layers):
            W_key = f'W{i}'
            b_key = f'b{i}'
            if W_key in initial_params and b_key in initial_params:
                W = initial_params[W_key]
                b = initial_params[b_key]
            else:
                raise ValueError(f"Weights or biases not found for layer {i}.")

            params.append({'W': W, 'b': b})
        return params

    def get_activation(self, activation_name):
        if activation_name == "ReLU":
            return self.ReLU
        elif activation_name == "softmax":
            return self.softmax
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")

    def ReLU(self, x):
        return np.maximum(0, x)

    def ReLU_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        y_true_classes = np.argmax(y_true, axis=1)
        log_likelihood = -np.log(y_pred[range(m), y_true_classes])
        loss = np.sum(log_likelihood) / m
        return loss

    def one_hot_encode(self, y, num_classes=4):
        one_hot = np.zeros((y.shape[0], num_classes))
        y = y.astype(int)
        one_hot[np.arange(y.shape[0]), y.squeeze()] = 1
        return one_hot

    def forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.layers)):
            W = self.params[i]['W']
            b = self.params[i]['b']
            z = np.dot(activations[-1], W) + b
            a = self.activations[i](z)
            activations.append(a)
        return activations

    def back_propagation(self, X, y_true, activations):
        m = X.shape[0]
        grads = []

        dz = activations[-1] - y_true
        for i in reversed(range(len(self.layers))):
            W = self.params[i]['W']
            a_prev = activations[i]
            dW = np.dot(a_prev.T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            grads.insert(0, {'dW': dW, 'db': db})
            
            if i > 0:  # Don't backpropagate after the first layer
                dz = np.dot(dz, W.T) * self.ReLU_derivative(activations[i])

        return grads

    def update_params(self, grads, learning_rate):
        for i in range(len(self.layers)):
            self.params[i]['W'] -= learning_rate * grads[i]['dW']
            self.params[i]['b'] -= learning_rate * grads[i]['db']

   
    def calculate_accuracy(self, X, y_true):
        activations = self.forward_propagation(X)
        predictions = np.argmax(activations[-1], axis=1)
        accuracy = np.mean(predictions == y_true.squeeze()) * 100
        return accuracy

    def train(self, X_train, y_train, X_test, y_test, learning_rate, epochs):
        y_train_one_hot = self.one_hot_encode(y_train)
        y_test_one_hot = self.one_hot_encode(y_test)

        train_costs, test_costs = [], []
        train_accuracies, test_accuracies = [], []

        for epoch in range(epochs):
            activations = self.forward_propagation(X_train)
            grads = self.back_propagation(X_train, y_train_one_hot, activations)
            self.update_params(grads, learning_rate)

            # Forward pass for test set
            test_activations = self.forward_propagation(X_test)

            # Compute loss and accuracy
            train_loss = self.cross_entropy_loss(activations[-1], y_train_one_hot)
            test_loss = self.cross_entropy_loss(test_activations[-1], y_test_one_hot)

            train_costs.append(train_loss)
            test_costs.append(test_loss)

            train_accuracy = self.calculate_accuracy(X_train, y_train)
            test_accuracy = self.calculate_accuracy(X_test, y_test)

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")
                print(f"Train Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}%")

            # save_gradients_to_csv(grads)

        return train_costs, test_costs, train_accuracies, test_accuracies

# Define the layers
layers = [
    {"neurons": 100, "activation": "ReLU"},
    {"neurons": 40, "activation": "ReLU"},
    {"neurons": 4, "activation": "softmax"}
]

# Create the network
network = Network(layers, initial_params)

# Function to plot graphs for each learning rate
def plot_graphs(train_costs, test_costs, train_accuracies, test_accuracies, learning_rate):
    epochs_range = range(len(train_costs))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_costs, label='Train Cost')
    plt.title(f"Train Cost vs Epochs (Learning Rate = {learning_rate})")
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

    # Plot testing cost
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, test_costs, label='Test Cost')
    plt.title(f"Test Cost vs Epochs (Learning Rate = {learning_rate})")
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy', linestyle='--')
    plt.title(f"Accuracy vs Epochs (Learning Rate = {learning_rate})")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

learning_rates = [1, 0.1, 0.001]
epochs = 1000

for lr in learning_rates:
    print(f"\nTraining with learning rate = {lr}")
    train_costs, test_costs, train_accuracies, test_accuracies = network.train(X_train, y_train, X_test, y_test, lr, epochs)
    plot_graphs(train_costs, test_costs, train_accuracies, test_accuracies, lr)







