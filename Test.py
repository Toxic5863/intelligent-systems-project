from sklearn.neural_network import MLPClassifier
import numpy as np
import GANetworks

training_input_list = np.zeros((1000, 2))
for i in range(training_input_list.shape[0]):
    training_input_list[i][0] = np.random.randint(0, 2)
    training_input_list[i][1] = np.random.randint(0, 2)
training_xor_list = np.zeros((training_input_list.shape[0], 1))
for i in range(training_xor_list.shape[0]):
    training_xor_list[i][0] = (training_input_list[i][0] != training_input_list[i][1])

# print(training_input_list[:3])
# print(training_xor_list[:3])

print(GANetworks.startGenetics(training_input_list, training_xor_list))
