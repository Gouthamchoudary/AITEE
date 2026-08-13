import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def main():
    # input 1
    x1 = np.array([[-0.25],
                   [0.10], 
                   [-0.25]])

    # desired output 1
    d1= np.array([[0.95]])

    # input 2
    x2 = np.array([[0.15],
                   [0.30],
                   [0.1]])

    # desired output 2
    d2 = np.array([[0.90]])


    #learning rate
    n = 0.5

    # initial weights
    w = np.array()
if __name__ == "__main__":
    main()