# Name: Ismail Al Abdali
# Course: CS460G
# Instructor: Dr.Brent Harrison
import pandas as pd
import numpy as np
from Neural_Network_model import train_neural_network, model_predict,accuracy_score

# get our data ready 
train_set = pd.read_csv("./data/mnist_train_0_1.csv",header = None)
test_set = pd.read_csv("./data/mnist_test_0_1.csv",header = None)

# get the labels from the first column of the data frame which are labels, and other remaining data will be a x training set
x_train ,y_train = train_set.drop(0, axis = 1),train_set.iloc[:,0] 

# get the x_test data and y_test labels
x_test ,y_test = test_set.drop(0, axis = 1),test_set.iloc[:,0]

# Now that we've got our data lets fit our model
# initialize the layers of our model:
net_shapes_dict = {
    "input_layer_shape" : (784,1),
    "hidden_layer_shape" : (784,6),
    "hidden_to_out_shape" : (1,6),
    "output_layer_shape" : (1,1)
    }
# fit our model with our training set
params = train_neural_network(net_shapes_dict,x_train,y_train,alpha = 1,bias = 1) # optimal alpha and bias
### using our training set to get the accuracy 

# get predictedtion labels for train set
y_pred = model_predict(params,x_train)
# get accuracy score for train set
score = accuracy_score(y_train,y_pred)
print("Neural Network Accuracy Score {}% on Trainset".format(round(score*100,2)))


# using our testset to get our model accuracy 

# get predictedtion labels 
y_pred = model_predict(params,x_test)
# get model score on testset
score = accuracy_score(y_test,y_pred)
print("Neural Network Accuracy Score {}% on Testset".format(round(score*100,2)))


### this code used for model optimization:
# after optimizaing our model best alpha (learning rate) is 1.0 and bias with 1.0

"""
for i in np.arange(0.1,1.1,0.1):
    i = round(i,2)
    for j in np.arange(0.1,1.1,0.1):
        j = round(j,2)
        params = train_neural_network(net_shapes_dict,x_train,y_train,alpha = i,bias = j) # optimal alpha and bias
        y_pred = model_predict(params,x_test)
        score = accuracy_score(y_test,y_pred)
        print("Accuracy {} when alpha {} and bias {}".format(round(score*100,2),i,j))
    """