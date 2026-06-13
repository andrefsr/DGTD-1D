import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

def pontos_gll(N):
    """
    Retorna os pontos GLL e seus respectivos pesos para um polinômio de ordem N
    no intervalo de referência [-1, 1].
    """
    if N == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    
    # Os pontos internos GLL são as raízes do polinômio de Jacobi P_{N-1}^{(1,1)}
    pontos_internos, _ = sp.roots_jacobi(N - 1, 1, 1)
    
    # Adiciona as extremidades -1 e 1
    pontos = np.concatenate(([-1.0], pontos_internos, [1.0]))
    
    # Calcule o peso das extremidades
    peso_borda = 2.0 / (N * (N + 1))

    # Calcule os pesos apenas para os pontos internos
    pesos_internos = 2.0 / (N * (N + 1) * (sp.eval_legendre(N, pontos_internos)) ** 2)

    # Junta tudo em um único vetor estático
    pesos = np.concatenate(([peso_borda], pesos_internos, [peso_borda]))
    
    return pontos, pesos

def gerar_malha_gll_1d(X_min, X_max, num_elementos, ordem_N):
    """
    Gera uma malha 1D para DGTD estruturada em matriz (Np x num_elementos)
    """
    Np = ordem_N + 1
    xi, _ = pontos_gll(ordem_N) # Array de tamanho (Np,) contendo os nós locais [-1, 1]
    
    fronteiras_elementos = np.linspace(X_min, X_max, num_elementos + 1)
    
    # 1. Criamos uma matriz de ZEROS com o tamanho exato do DG: (Np, num_elementos)
    malha_global = np.zeros((Np, num_elementos))
    
    for i in range(num_elementos):
        x_esq = fronteiras_elementos[i]
        x_dir = fronteiras_elementos[i+1]
        
        # Mapeamento linear (Fica igual ao seu)
        pontos_elemento = x_esq + (xi + 1.0) * (x_dir - x_esq) / 2.0
        
        # 2. Em vez de dar extend, salvamos o elemento INTEIRO na coluna 'i' da matriz
        # (Mantendo o primeiro e o último ponto sem picotar nada!)
        malha_global[:, i] = pontos_elemento
        
    return malha_global, fronteiras_elementos

def vis_grade(X_inicio,X_fim,n_elementos,ordem_polinomio):
    # Gerar a malha corrigida
    malha, nos_macro = gerar_malha_gll_1d(X_inicio, X_fim, n_elementos, ordem_polinomio)

    # --- Visualização da Malha ---
    plt.figure(figsize=(12, 3))
    plt.scatter(malha, np.zeros_like(malha), color='blue', s=40, zorder=3, label='Pontos GLL')

    for no in nos_macro:
        plt.axvline(x=no, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=nos_macro[0], color='red', linestyle='--', alpha=0.7, label='Fronteira dos Elementos')

    plt.title(f"Malha 1D GLL Corrigida ({n_elementos} elementos, Polinômios de Ordem {ordem_polinomio})")
    plt.xlabel("Coordenada X")
    plt.yticks([])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return