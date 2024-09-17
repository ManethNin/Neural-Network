print("hello")
import pandas as pd
import numpy as np

wo = pd.read_csv("C:/Users/REDTECH/Desktop/DNN Assignment/Assignment_1/Task_1/a/w.csv", header=None)
b0 = pd.read_csv("C:/Users/REDTECH/Desktop/DNN Assignment/Assignment_1/Task_1/a/b.csv", header=None)
# X_train = pd.read_csv(r"C:\Users\REDTECH\Desktop\DNN Assignment\Assignment_1\Task_2\x_train.csv", header=None)
# y_train = pd.read_csv(r"C:\Users\REDTECH\Desktop\DNN Assignment\Assignment_1\Task_2\y_train.csv", header=None)

weights_btw_layer0_to_layer1 = wo.iloc[:14, 1:].values  #(14, 100)
bias_for_layer1 = b0.iloc[:1, 1:].values  #(100,)

weights_btw_layer1_to_layer2 = wo.iloc[14:114, 1:41].values   #100,40
bias_for_layer2 = b0.iloc[1:2,1:41].values  #(40,)

weights_btw_layer2_to_layer3 = wo.iloc[114:, 1:5].values   #40,4
bias_for_layer3 = b0.iloc[2:,1:5].values  #(4,)

initial_params = weights_btw_layer0_to_layer1, bias_for_layer1, weights_btw_layer1_to_layer2, bias_for_layer2, weights_btw_layer2_to_layer3, bias_for_layer3

def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    x = np.array(x, dtype=float)
    max_x = np.amax(x, 1).reshape(x.shape[0],1) 
    e_x = np.exp(x - max_x ) 
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    epsilon = 1e-12  # Small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    y_true_classes = np.argmax(y_true, axis=1)  # Convert one-hot to class labels
    log_likelihood = -np.log(y_pred[range(m), y_true_classes])
    loss = np.sum(log_likelihood) / m
    return loss

def one_hot_encode(y, num_classes=4):
    one_hot = np.zeros((y.shape[0], num_classes))
    y = y.astype(int)
    one_hot[np.arange(y.shape[0]), y.squeeze()] = 1
    return one_hot

def update_params(params, grads, learning_rate):
    W1, b1, W2, b2, W3, b3 = params
    dW1, db1, dW2, db2, dW3, db3 = grads
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    
    return W1, b1, W2, b2, W3, b3

# For testing
X=[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]
X = pd.DataFrame(X)
X_train = X.T
y_train = [3]
y_train = pd.DataFrame(y_train)


def forward_propagation(X,params):
    w1, b1, w2, b2, w3, b3 = params
    #Layer 1
    z1 = np.dot(X,w1) + b1    #(1,100)
    h1 = ReLU(z1)   #(1,100)    

    #Layer 2
    z2 = np.dot(h1, w2) + b2
    h2 = ReLU(z2)    #1,40

    #Layer 3
    z3 = np.dot(h2, w3) + b3
    h3 = softmax(z3)    #1,4

    return z1, h1, z2, h2, z3, h3



def back_propagation(X, y_true, params, activations):
    W1, b1, W2, b2, W3, b3 = params
    z1, h1, z2, h2, z3, h3 = activations
    
    m = X.shape[0]
    # Output layer error
    dz3 = h3 - y_true 
    dz3 /= m
    
    dW3 = np.dot(h2.T, dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)
    
    # Layer 2 error
    dz2 = np.dot(dz3, W3.T) * ReLU_derivative(z2)
    dW2 = np.dot(h1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    # Layer 1 error
    dz1 = np.dot(dz2, W2.T) * ReLU_derivative(z1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3

def train(X, y, params, epochs, learning_rate):
    y_one_hot = one_hot_encode(y)  # one-hot encoding

    for epoch in range(epochs):  
        activations = forward_propagation(X,params)  
        grads = back_propagation(X, y_one_hot, params, activations)  
        params = update_params(params, grads, learning_rate)  
        
        # Calculate and print the loss (how bad the guesses were)
        _, _, _, _, _, h3 = activations
        loss = cross_entropy_loss(h3, y_one_hot)
   
        print(f"Epoch {epoch}, Loss: {loss}")
    
    return grads 

trained_params = train(X_train, y_train, initial_params, epochs=1, learning_rate=0.1)

print("dw1", trained_params[0].shape)
print("db1", trained_params[1].shape)
print("dw2", trained_params[2].shape)
print("db2", trained_params[3].shape)
print("dw3", trained_params[4].shape)
print("db3", trained_params[5].shape)
