import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(y):
    return y*(1-y)

def run_1_epoch(w , patterns, eta):
    print("epoch =1")

    for i, (x,d) in enumerate(patterns, start=1):
        z=np.dot(w,x)

        y=sigmoid(z)

        error = d-y
        delta = error*sigmoid_derivative(y)

        dw=eta*(delta)*x
        w=w+dw

        print(f"Pattern{i}")
        print (f" z = {z:.5f}")
        print (f" y = {y:.5f}")
        print (f" weights= {np.round(w,5)} \n")

def run_100_epochs(w,patterns, eta, num_epochs=100):
    print(f"{num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        total_error = 0.0

        for x, d in patterns:
            z = np.dot(w, x)
            y = sigmoid(z)
            error = d - y
            delta = error * sigmoid_derivative(y)
            w = w + eta * delta * x
            total_error += abs(error)
            
        if epoch == 1 or epoch % 20 == 0 or epoch == num_epochs:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | Total Error: {total_error:.5f} | Weights: {np.round(w, 5)}"
            )
    print("\n")
            
def run_until_tolerance(w, patterns, eta, tol=0.001, max_epochs=10000):
    print("tolerance epochs")
    epoch =0

    while epoch < max_epochs:
        epoch += 1
        max_error = 0.0

        for x, d in patterns:
            z = np.dot(w, x)
            y = sigmoid(z)
            error = d - y
            delta = error * sigmoid_derivative(y)
            w = w + eta * delta * x

            if abs(error) > max_error:
                max_error = abs(error)

        if max_error < tol:
            break

    print(f"Converged in {epoch} epochs!")
    print(f"Final Max Error: {max_error:.5f}")
    print(f"Final Weights  : {np.round(w, 5)}\n")

eta = 0.5
w_initial = np.array([-0.3, 0.2, -0.1])

patterns = [
    (np.array([-0.25, 0.10, -0.20]), 0.95),
    (np.array([0.15, -0.30, 0.10]), 0.90),
]

run_1_epoch(w_initial.copy(), patterns, eta)

run_100_epochs(w_initial.copy(), patterns, eta, num_epochs=100)

run_until_tolerance(w_initial.copy(), patterns, eta, tol=0.001)
