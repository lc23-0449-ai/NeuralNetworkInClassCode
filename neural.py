#Author: Toby Strawser
import numpy as np
from math import exp

learning_rate = 0.01
inputs = (np.array([1, 0,0]),
           np.array([1, 0,1]),
           np.array([1, 1,0]),
           np.array([1, 1,1]))

teachers = (0, 1, 1, 1)


class Neuron:
    def __init__(self):
        self.weights = (np.random.random(3) -0.5) / 100
        self.activation = None
        self.delta = None

    def update_activation(self,inputs):
        self.activation = sigmoid(self.weights@input)

    def update_delta(self, teacher):
        a = self.activation
        t = teacher
        self.delta = -a * (1 - a) * (t - a)
        pass
    def update_weights(self, input):
        self.weights -= learning_rate * input * self.delta
        pass



def sigmoid(s):
    return 1 / (1 + exp(-s))


def train(neuron, input, teacher):
    #Run forward
    neuron.update_activation(input)

    #Calculate weight changes
    neuron.update_delta(teacher)

    #update weights
    neuron.update_weights(input)


neuron = Neuron()

for input in inputs:
    neuron.update_activation(input)
    print(f'{input} -> {neuron.activation}')

print('---')

for i in range(100000):

    for input, teacher in zip(inputs, teachers):
        train(neuron, input, teacher)

for input in inputs:
    print(f'{input} -> {neuron.activation}')