import numpy as np

def calcular_matriz_fluxo(u, a):
    """
    Calcula a matriz F (Np x K) de termos de fronteira usando fluxo Upwind (para a > 0)
    """
    Np, K = u.shape
    F = np.zeros_like(u)
    
    # --- FRONTEIRA ESQUERDA (x_L) de cada elemento ---
    u_esq_local = u[0, :]                          # u^- na esquerda
    u_esq_vizinho = np.roll(u[-1, :], 1)           # u^+ na esquerda (fim do elemento anterior)
    
    # No Upwind com a > 0, o fluxo numérico (au)* vem do vizinho da esquerda
    fluxo_num_esq = a * u_esq_vizinho
    fluxo_local_esq = a * u_esq_local
    
    # Primeira linha de F recebe o saldo da esquerda (com o sinal negativo da normal)
    F[0, :] = -1.0 * (fluxo_num_esq - fluxo_local_esq)
    
    # --- FRONTEIRA DIREITA (x_R) de cada elemento ---
    u_dir_local = u[-1, :]                         # u^- na direita
    u_dir_vizinho = np.roll(u[0, :], -1)           # u^+ na direita (inicio do elemento seguinte)
    
    # No Upwind com a > 0, o fluxo numérico (au)* vem de dentro do próprio elemento
    fluxo_num_dir = a * u_dir_local
    fluxo_local_dir = a * u_dir_local
    
    # Última linha de F recebe o saldo da direita (fluxo_num - fluxo_local)
    # Note que para a > 0, (fluxo_num_dir - fluxo_local_dir) vai dar ZERO!
    # Isso é fisicamente correto: a onda sai do elemento sem gerar resistência interna na borda de saída.
    F[-1, :] = 1.0 * (fluxo_num_dir - fluxo_local_dir)
    
    return F
