# from scipy.io import arff

import arff # https://github.com/renatopp/liac-arff
import numpy as np
from dnn_app_utils_v3 import *

def main():
    ## bkp
    # decoder = arff.ArffDecoder()
    # data = decoder.decode(file, encode_nominal=True)

    # np_arr_flag = np_arr_trans[41]
    # np_arr_flag = np_arr_flag.reshape((np_arr_flag.shape[0], 1)).T

    np_arr_train, np_arr_train_flag, np_arr_test, np_arr_test_flag = prepare_data()
    np_arr_train_norm = normalize_data_set(np_arr_train)
    np_arr_test_norm = normalize_data_set(np_arr_test)

    layers_dims = define_dims_two_layer()
    parameters = two_layer_model(np_arr_train_norm, np_arr_train_flag, layers_dims, num_iterations = 3000, print_cost=True)
    predictions_train = predict(np_arr_train_norm, np_arr_train_flag, parameters)
    predictions_test = predict(np_arr_test_norm, np_arr_test_flag, parameters)

    # layers_dims = define_dims_l_layer()
    # parameters = L_layer_model(np_arr_train_norm, np_arr_train_flag, layers_dims, num_iterations = 3500, print_cost = True)
    # pred_train = predict(np_arr_train_norm, np_arr_train_flag, parameters)
    # pred_test = predict(np_arr_test_norm, np_arr_test_flag, parameters)

def prepare_data():
    print("prepare data: ")
    train_file = open('D:/Styding/Neural networks/NSLKDD-Dataset-arff/DOS -d/KDDTrain20DOS.arff', 'rb')
    test_file = open('D:/Styding/Neural networks/NSLKDD-Dataset-arff/DOS -d/KDDTest21DOS.arff', 'rb')
    train_data = arff.load(train_file)
    test_data = arff.load(test_file)

    np_arr_train = np.array(train_data['data'])
    np_arr_train, np_arr_train_flag = np.vsplit(np_arr_train.T, [41]) # last column - attack/not attack flag
    print("arr train shape - " + str(np_arr_train.shape))
    print("arr train flag shape - " + str(np_arr_train_flag.shape))
    np_arr_train = np_arr_train.astype(float)
    np_arr_train_flag = np_arr_train_flag.astype(float)
    ## Sanity check
    # print(np_arr_train_flag[0, 0:15])
    # print(np_arr_train[39, 0:15])

    np_arr_test = np.array(test_data['data'])
    np_arr_test, np_arr_test_flag = np.vsplit(np_arr_test.T, [41]) # last column - attack/not attack flag
    print("arr test shape - " + str(np_arr_test.shape))
    print("arr test flag shape - " + str(np_arr_test_flag.shape))
    np_arr_test = np_arr_test.astype(float)
    np_arr_test_flag = np_arr_test_flag.astype(float)

    return np_arr_train, np_arr_train_flag, np_arr_test, np_arr_test_flag

# parform min-max normalization
def normalize_data_set(data_arr):
    for i in range(data_arr.shape[0]):
        r = data_arr[i]
        min_val = r.min()
        max_val = r.max()
        # print("i = " + str(i) + " , min = " + str(min_val) + " , max = " + str(max_val))
        if min_val > 0 or max_val > 1:
            data_arr[i] = (r - min_val) / (max_val - min_val)
    # print("after normalization: ")
    # for i in range(data_arr.shape[0]):
    #     r = data_arr[i]
    #     min_val = r.min()
    #     max_val = r.max()
    #     print("i = " + str(i) + " , min = " + str(min_val) + " , max = " + str(max_val))
    return data_arr

def define_dims_two_layer():
    ### CONSTANTS DEFINING THE MODEL ####
    n_x = 41
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    return layers_dims

def define_dims_l_layer():
    ### CONSTANTS ###
    layers_dims = [41, 29, 17, 10, 5, 1] #  5-layer model
    return layers_dims

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3500, print_cost=False):
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    print("m - " + str(m))
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range (0, num_iterations):
        # forward prop
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        # compute cost
        cost = compute_cost(A2, Y)
        # init backward prop
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        # backward prop
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    np.random.seed(1)
    costs = []                         # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        # Compute cost.
        cost = compute_cost(AL, Y)
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    # # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    return parameters

if __name__ == '__main__':
    main()
