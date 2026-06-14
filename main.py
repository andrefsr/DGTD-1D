import numpy as np
import matplotlib.pyplot as plt

import gll1D as gll 
import fluxo_numerico as flux
import operadores as op 

N = 4
K = 100
Nfaces = 2 
Nfp = 1

Np = N + 1

r, VX = gll.gerar_malha_gll_1d(0,1,K,N)
nos_xi, _ = gll.pontos_gll(N)

V =  op.V1D(N,nos_xi)
invV = np.linalg.inv(V)

Dr = op.matriz_diferenciacao(N,nos_xi,V)

LIFT = flux.Lift1D(Np,V)

v = flux.EtoV(K)
va = v[:,0]
vb = v[:,1]
x = np.ones((Np,1))*VX[va] + 0.5*(r+1)*(VX[vb] - VX[va])

rx, J = flux.fator_geometrico1D(x,Dr)

NODETOL = 1e-10
fmask1 = np.where(np.abs(nos_xi + 1) < NODETOL)[0]
fmask2 = np.where(np.abs(nos_xi - 1) < NODETOL)[0]
Fmask = np.vstack((fmask1, fmask2)).T
Fx = x[Fmask.flatten(), :]

nx = flux.normal_1D(K)
Fscale = 1/(J[Fmask,:])

etoe, etof = flux.Connect1D(flux.EtoV(K))

vmapM, vmapP, vmapB, mapB = flux.BuildMaps1D(Np,K,r,etoe,etof)

def advecrhs1D(u,time,a,K):
    alpha = 1
    du = np.zeros((Nfp*Nfaces,K))
    
    u_flat = u.flatten(order='F')

    du = (
        (u_flat[vmapM] - u_flat[vmapP])
        * (
            a*nx.flatten(order='F')
            - (1-alpha)*np.abs(a*nx.flatten(order='F'))
        )
        / 2.0
    )

    uin = -np.sin(a*time)

    flux_term = (Fscale.flatten(order='F') * du).reshape((Nfp*Nfaces, K), order='F')

    rhsu = -a*rx*(Dr @ u) + LIFT @ flux_term

    return rhsu

def advec1D(u,FinalTime):
    time = 0

    resu = np.zeros((Np,K))

    xmin = np.min(np.abs(x[0,:] - x[1,:]))
    CFL = 0.75
    dt = CFL/(2*np.pi)*xmin
    dt = .5*dt
    Nsteps = int(np.ceil(FinalTime/dt))
    dt = FinalTime/Nsteps

    a = 2*np.pi

    resu = np.zeros_like(u)

    rk4a = np.array([0.0, -567301805773.0/1357537059087.0, -2404267990393.0/2016746695238.0, -3550918686646.0/2091501179385.0, -1275806237668.0/842570457699.0])
    rk4b = np.array([1432997174477.0/9575080441755.0, 5161836677717.0/13612068292357.0, 1720146321549.0/2090206949498.0, 3134564353537.0/4481467310338.0, 2277821191437.0/14882151754819.0])
    rk4c = np.array([0.0, 1432997174477.0/9575080441755.0, 2526269341429.0/6820363962896.0, 2006345519317.0/3224310063776.0, 2802321613138.0/2924317926251.0])

    for intrk in range(5):

        timelocal = time + rk4c[intrk]*dt

        rhsu = advecrhs1D(u,timelocal,a,K)

        resu = rk4a[intrk]*resu + dt*rhsu

        u = u + rk4b[intrk]*resu

    time += dt
    return u

u = np.sin(x)
FinalTime = 1000

u = advec1D(u,FinalTime)

plt.figure()
plt.title(f'DGTD 1D \n K = {K} e N = {N}')
plt.ylabel(r'$u_h(x)$')
plt.xlabel('x')
plt.grid(alpha=0.3)
#plt.plot(r, u, color='royalblue')#, label=f'$t_f$ = {Nt*dt:.4f}')
plt.plot(x.flatten(order='F'),u.flatten(order='F'))
#plt.legend()
plt.show()
