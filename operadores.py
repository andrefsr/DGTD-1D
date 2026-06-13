import numpy as np

def polinomio_legendre(i, xi):
    if i == 0: return np.ones_like(xi)
    if i == 1: return xi
    P_0 = np.ones_like(xi)
    P_1 = xi
    for n in range(1, i):
        P_next = ((2 * n + 1) * xi * P_1 - n * P_0) / (n + 1)
        P_0, P_1 = P_1, P_next
    return P_1

def V1D(N, nos_xi):
    Np = N + 1
    V = np.zeros((Np, Np))
    for j in range(Np):
        V[:, j] = polinomio_legendre(j, nos_xi)
    return V

def derivada_legendre(i, xi):
    if i == 0: 
        return np.zeros_like(xi)
    if i == 1:
        return np.ones_like(xi)
        
    dP = np.zeros_like(xi)
    
    nas_bordas = np.isclose(np.abs(xi), 1.0)
    no_interior = ~nas_bordas
    
    dP[no_interior] = (i * (polinomio_legendre(i-1, xi[no_interior]) - xi[no_interior] * polinomio_legendre(i, xi[no_interior]))) / (1.0 - xi[no_interior]**2)
    
    valor_borda = i * (i + 1) / 2.0
    dP[nas_bordas] = np.where(xi[nas_bordas] > 0, valor_borda, ((-1)**(i-1)) * valor_borda)
    
    return dP

def matriz_diferenciacao(N, nos_xi, V):
    Np = N + 1
    Vx = np.zeros((Np, Np))
    for j in range(Np):
        Vx[:, j] = derivada_legendre(j, nos_xi)
    
    D = np.dot(Vx, np.linalg.inv(V))
    return D
