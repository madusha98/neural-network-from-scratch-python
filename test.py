from nn import NerualNetwork
import random
from utils import printProgressBar


training_data = [
  {
    "inputs": [0, 1],
    "targets": [1],
  },
  {
    "inputs": [1, 0],
    "targets": [1],
  },
  {
    "inputs": [0, 0],
    "targets": [0],
  },
  {
    "inputs": [1, 1],
    "targets": [0],
  }
]

ITERATIONS = 50000
LEARNING_RATE = 0.1

def testXOR():

    nn = NerualNetwork(2,3,1, LEARNING_RATE)

    # show a progress bar
    printProgressBar(0, ITERATIONS, prefix = 'Progress:', suffix = 'Complete | 0 Iterations', length = 50)

    for i in range(ITERATIONS):

        printProgressBar(i + 1, ITERATIONS, prefix = 'Progress:', suffix = 'Complete | ' + str(i + 1) + ' Iterations', length = 50)

        data = random.choice(training_data)
        nn.train(data['inputs'], data['targets'])
    
    print(nn.feedforward([0,0]))
    print(nn.feedforward([0,1]))
    print(nn.feedforward([1,0]))
    print(nn.feedforward([1,1]))

testXOR()