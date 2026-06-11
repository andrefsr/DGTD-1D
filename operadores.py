import numpy as np

def polinomio_legendre(i, xi):
    """Calcula o i-ésimo polinômio de Legendre no ponto xi"""
    # Implementação da fórmula de recorrência de Bonnet
    if i == 0: return np.ones_like(xi)
    if i == 1: return xi
    P_0 = np.ones_like(xi)
    P_1 = xi
    for n in range(1, i):
        P_next = ((2*n + 1)*xi*P_1 - n*P_0) / (n + 1)
        P_0, P_1 = P_1, P_next
    return P_1

def matriz_vandermonde(N, nos_xi):
    """Gera a matriz de Vandermonde Modal-Nodal"""
    Np = N + 1
    V = np.zeros((Np, Np))
    for j in range(Np):
        V[:, j] = polinomio_legendre(j, nos_xi)
    return V

def derivada_legendre(i, xi):
    """Calcula a derivada do i-ésimo polinômio de Legendre"""
    if i == 0: return np.zeros_like(xi)
    # Relação clássica: a derivada depende do polinômio de ordem anterior
    return (i * (polinomio_legendre(i-1, xi) - xi * polinomio_legendre(i, xi))) / (1 - xi**2 + 1e-15)

def matriz_diferenciacao(N, nos_xi, V):
    """Gera a matriz D que calcula as derivadas em cima dos nós"""
    Np = N + 1
    Vx = np.zeros((Np, Np))
    for j in range(Np):
        Vx[:, j] = derivada_legendre(j, nos_xi)
    
    # D = Vx * inv(V)
    D = np.dot(Vx, np.linalg.inv(V))
    return D