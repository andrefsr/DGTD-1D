import numpy as np

def lsrk4_passo(U, residue_function, dt):

    a = [0.0, -5.0 / 9.0, -153.0 / 128.0, -2.0 / 3.0, -1.0 / 4.0]
    b = [1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0, 2.0 / 3.0, 1.0 / 6.0]

    dW = np.zeros_like(U)

    for m in range(5):
        R = residue_function(U)

        dW = a[m] * dW + dt * R
        U = U + b[m] * dW

    return U