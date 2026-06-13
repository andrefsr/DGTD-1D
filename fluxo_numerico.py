import numpy as np

Nfaces = 2
Nfp = 1

def Lift1D(Np,V):
    Emat = np.zeros((Np,Nfaces*Nfp))
    Emat[0,0] = 1.
    Emat[Np - 1,1] = 1.

    LIFT = V @ (V.T @ Emat)
    
    return LIFT

def fator_geometrico1D(x,Dr):
    xr = Dr @ x
    J = xr
    rx = 1./J
    return rx , J

def normal_1D(K):
    nx = np.ones(Nfp*Nfaces,K)
    nx[0,:] = -1
    return nx

def EtoV(K):

    EToV = np.zeros((K,2), dtype=int)

    for k in range(K):
        EToV[k,:] = [k, k+1]

    return EtoV


import numpy as np

def Connect1D(EToV):
    """
    EToV: matriz (K,2)
          Elemento -> vértices

    Retorna:
        EToE : Elemento -> elemento vizinho
        EToF : Elemento -> face do vizinho
    """

    Nfaces = 2

    # número de elementos
    K = EToV.shape[0]

    TotalFaces = Nfaces * K
    Nv = K + 1

    # face local -> vértice local
    vn = np.array([0, 1])

    # Face global -> Vértice global
    SpFToV = np.zeros((TotalFaces, Nv), dtype=int)

    sk = 0
    for k in range(K):
        for face in range(Nfaces):

            vertice = EToV[k, vn[face]]

            # EToV do MATLAB começa em 1
            SpFToV[sk, vertice] = 1

            sk += 1

    # Face global -> Face global
    SpFToF = SpFToV @ SpFToV.T - np.eye(TotalFaces, dtype=int)

    faces1, faces2 = np.where(SpFToF == 1)

    element1 = faces1 // Nfaces
    face1    = faces1 % Nfaces

    element2 = faces2 // Nfaces
    face2    = faces2 % Nfaces

    # Inicialização
    EToE = np.tile(np.arange(K)[:, None], (1, Nfaces))
    EToF = np.tile(np.arange(Nfaces), (K, 1))

    # Preenche conectividades
    EToE[element1, face1] = element2
    EToF[element1, face1] = face2

    return EToE, EToF


K = 4

EToE, EToF = Connect1D(EtoV(K))

#print(etov)
print(EToE)
print(EToF)