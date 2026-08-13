import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():

    # input vector
    x = np.array([[0.2], 
                  [0.4],
                  [0.5]])


    # weights
    w = np.array([[0.2, 0.1, -0.2],
              [-0.4, -0.35, 0.2],
              [-0.4, 0.4, -0.4]])

    #calculate Net
    net = w.T @x

    #output
    output =sigmoid(net)




    # Clean formatted printing using vectorized operations
    print("\n Net Inputs (z): ---")
    print(np.round(net.ravel(), 5))

    print("\n \nActivation Outputs (y): ---")
    print(np.round(output.ravel(), 5))


if __name__ == "__main__":
    main()
