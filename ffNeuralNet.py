import numpy as np


# class that creates a feedforward network object
class ffNet:
    def __init__(self, hmHiddenNeurons, hmIn, hmOut, config):
        self.inputs = np.empty(hmIn)  # inputs are [x u], column vector
        self.outputs = np.empty(hmOut)  # column vector of outputs: x
        self.V = np.random.uniform(low=config["wInitMin"], high=config["wInitMax"],
                                   size=(hmHiddenNeurons, hmIn))
        self.W = np.random.uniform(low=config["wInitMin"], high=config["wInitMax"],
                                   size=(hmOut, hmHiddenNeurons))

    def respond(self, inputs):
        """ w Abdollahimi inputs are [x, u]"""
        return self.wRespond(self.vRespond(inputs))

    def vRespond(self, inputs):
        try:
            self.vOutput = 2 / (1+np.exp(-2*np.matmul(self.V, inputs))) - 1
        except ValueError as e:
            print(type(e))
            print(e.args)
            print(e)
        return self.vOutput

    def wRespond(self, inputs):
        return np.matmul(self.W, inputs)

    def actFunDerivative(self, x):
        foo = np.exp(-2*x)
        return 4*foo/(1+foo)**2
