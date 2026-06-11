import numpy as np

def polinomio_legendre(i, xi):
    """Calcula o i-ésimo polinômio de Legendre no ponto xi"""
    # Implementação da fórmula de recorrência de Bonnet
    if i == 0: return np.ones_like(xi)
    if i == 1: return xi
    P_0 = np.ones_like(xi)
    P_1 = xi
    # Correção sutil no loop de recorrência de Bonnet
    for n in range(1, i):
        # n representa o polinômio atual P_1 (de ordem n)
        # P_0 é de ordem n-1
        P_next = ((2 * n + 1) * xi * P_1 - n * P_0) / (n + 1)
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
    """Calcula a derivada do i-ésimo polinômio de Legendre sem divisão por zero"""
    if i == 0: 
        return np.zeros_like(xi)
    if i == 1:
        return np.ones_like(xi)
        
    # Inicializa o vetor de derivadas
    dP = np.zeros_like(xi)
    
    # Máscaras para identificar onde xi está nas bordas ou no interior
    nas_bordas = np.isclose(np.abs(xi), 1.0)
    no_interior = ~nas_bordas
    
    # Cálculo para pontos internos (sua fórmula original sem risco de quebrar)
    dP[no_interior] = (i * (polinomio_legendre(i-1, xi[no_interior]) - xi[no_interior] * polinomio_legendre(i, xi[no_interior]))) / (1.0 - xi[no_interior]**2)
    
    # Valores analíticos exatos nas bordas para evitar divisão por zero
    valor_borda = i * (i + 1) / 2.0
    dP[nas_bordas] = np.where(xi[nas_bordas] > 0, valor_borda, ((-1)**(i-1)) * valor_borda)
    
    return dP

def matriz_diferenciacao(N, nos_xi, V):
    """Gera a matriz D que calcula as derivadas em cima dos nós"""
    Np = N + 1
    Vx = np.zeros((Np, Np))
    for j in range(Np):
        Vx[:, j] = derivada_legendre(j, nos_xi)
    
    # D = Vx * inv(V)
    D = np.dot(Vx, np.linalg.inv(V))
    return D