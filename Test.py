#Authors: Alexander Hewitt, Maranda Daughtery, Patrick Dooley
#Purpose: To test the GANetworks module and pickle the resulting neural network
#Date completed: April 30, 2020


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np
import pickle
import GANetworks


# -----------------creating the data with which GANetworks will be tested-------------------
training_input_list = np.zeros((1000, 2))
for i in range(training_input_list.shape[0]):
    training_input_list[i][0] = np.random.randint(0, 2)
    training_input_list[i][1] = np.random.randint(0, 2)
training_xor_list = np.zeros((training_input_list.shape[0], 1))
for i in range(training_xor_list.shape[0]):
    training_xor_list[i][0] = (training_input_list[i][0] != training_input_list[i][1])


# ----------------------running GANetworks for the data------------
GANetworks.startGenetics(training_input_list, training_xor_list)


# --------------------testing the network pickled by GANetworks-----------------------------
try:
    ann = open("evolved_network", "rb")
except:
    print("Error opening MLPClassifier object from file")
network = pickle.load(ann)
print(network.predict([[0,0]]))
predict_test = network.predict(training_input_list)
print(classification_report(training_xor_list, predict_test))
