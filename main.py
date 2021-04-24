import datasets.mnist.mnist_loader as mnist
from nn import NerualNetwork
import random
from utils import printProgressBar
from random import randrange
import pickle
import signal
import sys
from multiprocessing import Process
import keyboard
import time


# mnist.init()

data = mnist.load()

# print(data[1][2])

ITERATIONS = 100000
LEARNING_RATE = 0.1

def trainModel():

    nn = NerualNetwork(784,64,10, LEARNING_RATE)

    # show a progress bar
    printProgressBar(0, ITERATIONS, prefix = 'Progress:', suffix = 'Complete | 0 Iterations', length = 50)

    for i in range(ITERATIONS):

        printProgressBar(i + 1, ITERATIONS, prefix = 'Progress:', suffix = 'Complete | ' + str(i + 1) + ' Iterations', length = 50)
        index = randrange(5999)
        inputs = [x/255 for x in data[0][index]]

        targets = data[1][index]
        t = []
        for j in range(10):
            if (j == targets):
                t.append(1)
            else:
                t.append(0)
        nn.train(inputs, t)
        if keyboard.is_pressed('x'):
                break

    with open('models/model1.pkl', 'wb') as output:
    
        pickle.dump(nn, output, pickle.HIGHEST_PROTOCOL)

def resumeTraining():
    with open('models/model1.pkl', 'rb') as input:
        nn = pickle.load(input)

     # show a progress bar
    printProgressBar(0, ITERATIONS, prefix = 'Progress:', suffix = 'Complete | 0 Iterations', length = 50)

    for i in range(nn.iterations, ITERATIONS):

        printProgressBar(i + 1, ITERATIONS, prefix = 'Progress:', suffix = 'Complete | ' + str(i + 1) + ' Iterations', length = 50)
        index = randrange(5999)
        inputs = [x/255 for x in data[0][index]]

        targets = data[1][index]
        t = []
        for j in range(10):
            if (j == targets):
                t.append(1)
            else:
                t.append(0)
        nn.train(inputs, t )
        if keyboard.is_pressed('x'):
                break

    with open('models/model1.pkl', 'wb') as output:
    
        pickle.dump(nn, output, pickle.HIGHEST_PROTOCOL)
    

def test():

    
    with open('models/model1.pkl', 'rb') as input:
        nn = pickle.load(input)

    print('Iterations : ', nn.iterations)
    correct = 0
    for i in range(1000):
        res = nn.feedforward([x/255 for x in data[2][i]])
        highest = 0
        digit = -1

        for r in range(len(res)):
            
            if (highest < res[r]):
                highest = res[r]
                digit = r
        
        if (digit == data[3][i]):
            correct += 1
        

        # print('expected : ', data[3][i], 'actual : ',  digit)
    print('Accuracy : ', str((correct / 1000) * 100) + '%' )




    
# trainModel()
# resumeTraining()
test()



