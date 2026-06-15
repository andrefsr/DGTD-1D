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
    nx = np.ones((Nfp*Nfaces,K))
    nx[0,:] = -1
    return nx

def EtoV(K):

    EToV = np.zeros((K,2), dtype=int)

    for k in range(K):
        EToV[k,:] = [k, k+1]

    return EToV

def Connect1D(EToV):
    Nfaces = 2

    K = EToV.shape[0]

    TotalFaces = Nfaces * K
    Nv = K + 1

    vn = np.array([0, 1])

    SpFToV = np.zeros((TotalFaces, Nv), dtype=int)

    sk = 0
    for k in range(K):
        for face in range(Nfaces):

            vertice = EToV[k, vn[face]]

            SpFToV[sk, vertice] = 1

            sk += 1

    SpFToF = SpFToV @ SpFToV.T - np.eye(TotalFaces, dtype=int)

    faces1, faces2 = np.where(SpFToF == 1)

    element1 = faces1 // Nfaces
    face1    = faces1 % Nfaces

    element2 = faces2 // Nfaces
    face2    = faces2 % Nfaces

    EToE = np.tile(np.arange(K)[:, None], (1, Nfaces))
    EToF = np.tile(np.arange(Nfaces), (K, 1))

    EToE[element1, face1] = element2
    EToF[element1, face1] = face2

    return EToE, EToF

import numpy as np

def MeshGen1D(xmin, xmax, K):
    # Número de vértices
    Nv = K + 1

    # Coordenadas dos vértices
    VX = np.linspace(xmin, xmax, Nv)

    # Conectividade elemento-vértice
    EToV = np.column_stack([
        np.arange(K),
        np.arange(1, K + 1)
    ])

    return Nv, VX, K, EToV

def BuildMaps1D(Np, K, x, EToE, EToF):
    NODETOL = 1e-10

    Fmask = np.array([[0, Np-1]])
    #nodeids = np.arange(K*Np).reshape((Np, K), order='F')
    # O jeito CORRETO de gerar os IDs para mapeamento em Python DGTD
    nodeids = np.arange(K * Np).reshape((Np, K), order='F')

    vmapM = np.zeros((Nfp, Nfaces, K), dtype=int)
    vmapP = np.zeros((Nfp, Nfaces, K), dtype=int)

    for k1 in range(K):
        for f1 in range(Nfaces):

            vmapM[:, f1, k1] = nodeids[Fmask[:,f1], k1]

    for k1 in range(K):
        for f1 in range(Nfaces):

            k2 = EToE[k1,f1]
            f2 = EToF[k1,f1]

            vidM = vmapM[:,f1,k1]
            vidP = vmapM[:,f2,k2]

            x1 = x.flatten(order='F')[vidM]
            x2 = x.flatten(order='F')[vidP]

            D = (x1-x2)**2

            print("k1 =", k1, "f1 =", f1)
            print("vidM =", vidM)
            print("vidP =", vidP)
            print("x1 =", x1)
            print("x2 =", x2)
            print("D =", D)
            print()

            if np.all(D < NODETOL):
                vmapP[:,f1,k1] = vidP

    vmapM = vmapM.flatten(order='F')
    vmapP = vmapP.flatten(order='F')
    mapB = np.where(vmapM == vmapP)[0]
    vmapB = vmapM[mapB]
    return vmapM, vmapP, vmapB, mapB

