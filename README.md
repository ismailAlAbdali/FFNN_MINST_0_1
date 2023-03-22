# FFNN_MINST_0_1
FFNN: Building Simple Feet Forward Neural Network From scartch for 0_1 MINST DataSet
# How to Compile
1-	Make sure you have python 3.9+\
2-	Install the following libraries.\
a.	Pandas\
b.	NumPy\
You can Install them by typing in your terminal.\
-	pip install "name of library above"
3-	run the project using the command:\
-	python3 main.py

# Project files and breif decribtion of what they hold:

1-	Main.py:
 This file contains the main function of our project. First, it divides the trainset and testset into x and y and then fits the training set into our Neural Network model and finds the test set accuracy. \

2-	Neural_network_model.py: this file contains the feet forward neural network definition, and it contains the following functions:\
a.	Sigmoid(x) : sigmoid function\
b.	Sigmoid_der(x) : derivative of sigmoid 
c.	Error(u,g_in): gets the error compared to y_true value
d.	initialize_neural_network(hidden_layer_shape, hidden_to_out_connection_shape,output_layer_shape,bias) : this function initialize our neural network based on the given shape and it also initializes the bias term.
e.	forward_propagation(net_parameters,X): forward propagate the neural network and return both hidden layer deltas and output layer delta values.
f.	backPropagate_and_getUpdateWeights(net_parameters,forward_dict,X,y_true,alpha) : back propagate the neural network and use the returned values from forward propagation function to find best weights and bias in order to be updated. The function returns the weights and bias terms matrices so we can update them in the train_nerual_network function.
g.	train_neural_network(net_shapes,x_train,y_train,alpha = 1,bias = 1,random_State = 42): the main function to train or fit our model. It gets net_shapes which is the neural network architecture and takes the x_train and y_train parameters which is the data we want our model to train on. Also the function takes alpha(learning rate), bias, and random_State( to setup random seed), these three parameters are already initialized. Finally, the function returns the best parameters that can be used for predictions.
h.	model_predict(net_params,x_test): predicts test set values using the resulted neural network parameters from the train_neural_network function. 
i.	accuracy_score(y_true,y_pred): compares true labels with predicted labels and returns the model accuracy score.
Note: at the end of main.py file, there is some commented code that had been used to optimize alpha and bias values. The optimal values are currently based on train_neural_network function. 
