# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 14:58:25 2015

SIMULACION DE DISPOSITIVOS SEMICONDUCTORES - HETEROJUNTURAS de Al(x)Ga(1-x)As
by ARIEL CEDOLA

"""

import numpy as np
import matplotlib.pyplot as pyplot
from math import pi
import datetime

def lifetime(lifetimex0, nro_nodos):
    # Tiempo de vida media de los portadores en cada nodo de la malla
    taux = np.ones(nro_nodos)*lifetimex0
    return taux

def mobility(mobilityx0, vxsat, bx, array_campo_e):
    # Movilidad de los portadores en los puntos medios entre nodos de la malla
    movx = mobilityx0/((1+(mobilityx0*abs(array_campo_e)/vxsat)**bx)**(1/bx))
    return movx

def campo_electrico(array_potencial, grilla1D):
    # Calcula el campo eléctrico como la derivada del potencial
    campo_e = -np.diff(array_potencial)/grilla1D
    return campo_e

def bernu(x):
    # Calcula la función de Bernoulli
    x1 = -36.25
    x2 = -7.63e-6
    x3 = -x2
    x4 = 32.92
    x5 = 36.5
    y = np.zeros(np.shape(x)) #y = np.zeros(len(x))
    B1 = x<=x1
    y[B1] = -x[B1]
    B2 = (x>x1) & (x<x2)
    y[B2] = x[B2]/(np.exp(x[B2])-1+1.e-99)
    B3 = (x>=x2) & (x<=x3)
    y[B3] = 1-x[B3]/2
    B4 = (x>x3) & (x<x4)
    y[B4] = x[B4]*np.exp(-x[B4])/(1-np.exp(-x[B4])+1.e-99)
    B5 = (x>=x4) & (x<x5)
    y[B5] =  x[B5]*np.exp(-x[B5])
    y = -y
    return y
    
# DEFINICION DE ctes.
eps0 = 8.85e-14 # [F/cm]
q = 1.602e-19   # [coulombs] 
kb = 1.381e-23  # [Joules/K]
temp = 300.      # [K]
vt = kb*temp/q  # [volts]
h_planck = 6.626e-34    # [Joules.s]
c_light = 29979245800.   # [cm/s]
m0 = 9.1e-31 # [Kg]

# CARACTERISTICAS DEL DISPOSITIVO p-i-n A SIMULAR
# DEV1
#l_reg = np.array([50, 100, 1050, 300], float) # nm
#na_reg = np.array([0, 0, 0, 0], float) # cm-3
#nd_reg = np.array([1e15, 1e15, 1e15, 1e17], float) # cm-3
#Al_comp_reg = np.array([0, 0, 0, 0.3], float) # cm-3
# DEV2
l_reg = np.array([500, 500], float) # nm
#na_reg = np.array([0, 0], float) # cm-3 HORIO90
#nd_reg = np.array([1e15, 1e17], float) # cm-3 HORIO90
#Al_comp_reg = np.array([0., 0.25], float) # cm-3 HORIO90
#na_reg = np.array([0, 0], float) # cm-3 YANG93 FIG4
#nd_reg = np.array([1e15, 5e16], float) # cm-3 YANG93 FIG4
# Sección 3.1
#na_reg = np.array([0, 1e16], float) # cm-3 YANG93 FIG7b y 8
#nd_reg = np.array([1e16, 0], float) # cm-3 YANG93 FIG7b y 8
#Al_comp_reg = np.array([0.3, 0.], float) # cm-3
# Sección 3.2
#na_reg = np.array([1e17, 0], float) # cm-3 YANG93 FIG7b y 8
#nd_reg = np.array([0, 1e17], float) # cm-3 YANG93 FIG7b y 8
#Al_comp_reg = np.array([0., 0.25], float) # cm-3
# Sección 3.3
na_reg = np.array([0, 0], float) # cm-3 YANG93 FIG7b y 8
nd_reg = np.array([1e15, 1e15], float) # cm-3 YANG93 FIG7b y 8
Al_comp_reg = np.array([0., 0.25], float) # cm-3

# GEOMETRIA Y MALLA

# MALLA no uniforme DEV1
#x = np.r_[0:11:1, 12, 15:30:3, 30:71:1, 73, 76, \
#80:125:4, 127, 130:171:1, 173, 176, 180:1161:10, 1164, 1167, \
#1170:1231:1, 1235, 1240:1481:10, 1485, 1490:1501:1]*1e-7
# MALLA uniforme DEV2
x = np.r_[0:1001:1]*1e-7
 
celdas = len(x)-1
num_reg = len(l_reg)
long = sum(l_reg)*1e-7
centro  = np.argmin(abs(x-long/2)) # 395
h1i = x[1:]-x[:-1] 
h2i = np.insert((h1i[1:]+h1i[:-1])/2, 0, 1) # h2i = (h1i[1:]+h1i[:-1])/2

# SUNLIGHT
X_SUN = 1
SOLAR_GAAS = 1
REFLEX_GAAS = 0

# METODO NUMERICO
K_MAX = 100
K_LIMITE = 1500
POR_ITERACIONES = 1
tol_equ = 1e-7
tol = 1e-9#1e-7
DAMPING = 0.9

# PARAMETROS del material (GaAs)
eps_gaas = 13.18 # cte. dieléctrica PC1D
egap_gaas = 1.42 # Energy gap
aff_gaas = 4.06 # [eV]
eff_mass_n_gaas = 0.067
eff_mass_p_gaas = 0.48
tn0_gaas = 1e-9 # s
tp0_gaas = 1e-9 # s
c_dir_gaas = 2e-10 # Recombinación directa [cm3/seg]
cn_aug_gaas = 5e-30 # [cm6/seg]
cp_aug_gaas = 1e-31 # [cm6/seg]
movn0_gaas = 8000 # [cm2/v-seg] IOFFRE
movp0_gaas = 370 # [cm2/v-seg] IOFFRE
vnsat_gaas = 1.4e7 # [cm/seg]
vpsat_gaas = 8.37e6 # [cm/seg]
bn_gaas = 2
bp_gaas = 1

# RUTINA que determina el número de nodos que le corresponde a cada región
# nod_interfaces: array con los índices del último nodo de cada región
# nod_reg: cantidad de nodos de cada región, la suma dá celdas+1
nod_reg = np.zeros(num_reg, dtype=np.int32)
nod_interfaces = np.zeros(num_reg-1, dtype=np.int32)

if num_reg == 1:
    
    nod_reg[0]=celdas+1
    
else:
    
    nodo1 = -1
    for i in range(num_reg-1):
        nod_interfaces[i] = np.argmin(abs(x-np.sum(l_reg[0:(i+1)])*1e-7))
        nod_reg[i] = nod_interfaces[i] - nodo1
        nodo1 = nod_interfaces[i]
            
    nod_reg[num_reg-1] = celdas - nod_interfaces[num_reg-2]


# ARRAYS de doping densities y Al en cada nodo
na = np.zeros(celdas+1)
nd = np.zeros(celdas+1)
c = np.zeros(celdas+1)
Al_comp = np.zeros(celdas+1)

nod_interfaces2=np.r_[0, nod_interfaces+1, celdas+1]

for i in range(num_reg):
    np.put(na, np.r_[nod_interfaces2[i]:nod_interfaces2[i+1]],na_reg[i])
    np.put(nd, np.r_[nod_interfaces2[i]:nod_interfaces2[i+1]],nd_reg[i])
    np.put(Al_comp, np.r_[nod_interfaces2[i]:nod_interfaces2[i+1]],Al_comp_reg[i])

c = nd-na

# ARRAYS de bandgap, afinidad elctrónica, masas efectivas, cte dieléctrica,
# densidades efectivas de estados, tiempos de vida, movilidades
# IMPORTANTE: expresiones válidas para Al_comp<0.45
egap = egap_gaas + 1.247*Al_comp # [eV] YANG93
#affinity = aff_gaas - 0.6*(egap - egap_gaas) # YANG93
affinity = aff_gaas -1.1*Al_comp # IOFFRE = DINGLE RULE
eps = eps_gaas - 3*Al_comp # YANG93
eff_mass_n = (0.067 + 0.083*Al_comp)*m0 # [Kg] YANG93
eff_mass_p = (0.48 + 0.31*Al_comp)*m0 # [Kg] YANG93
NC = 1e-6*2*(2*pi*eff_mass_n*kb*temp/h_planck**2)**(3/2) # [cm-3]
NV = 1e-6*2*(2*pi*eff_mass_p*kb*temp/h_planck**2)**(3/2) # [cm-3]
movn = movn0_gaas - 2.2e4*Al_comp + 1e4*Al_comp**2 # IOFFRE
movp = movp0_gaas - 970*Al_comp + 740*Al_comp**2 # IOFFRE
tn = lifetime(tn0_gaas, len(x))
tp = lifetime(tp0_gaas, len(x))
nint = np.sqrt(NC*NV)*np.exp(-egap/2/vt) # Concentración intrínseca [cm-3]

# DENSIDADES de portadores en equilibrio en las regiones masivas
nint_reg = nint[nod_interfaces2[1:]-1]
c_reg = nd_reg - na_reg
mayori = (np.sqrt(np.power(c_reg, 2)+4*np.power(nint_reg, 2))+abs(c_reg))/2
minori = np.power(nint_reg, 2)/mayori
n_reg = (c_reg>=0)*mayori+(c_reg<0)*minori
p_reg = (c_reg<=0)*mayori+(c_reg>0)*minori

NC_reg = NC[nod_interfaces2[1:]-1]
NV_reg = NV[nod_interfaces2[1:]-1]
aff_reg = affinity[nod_interfaces2[1:]-1]

# POTENCIAL de contacto vbi (USO LISTA POR COMPRENSION)
vbi = np.asarray([aff_reg[i] - aff_reg[i+1] + \
vt*np.log(NC_reg[i]/NC_reg[i+1]*n_reg[i+1]/n_reg[i]) for i in range(num_reg-1)]) 

nod_ref = 0 # Nodo de la malla donde se toman los valores de referencia
aff_ref = affinity[nod_ref]
egap_ref = egap[nod_ref]
NC_ref = NC[nod_ref]
NV_ref = NV[nod_ref]
eps_ref = eps[nod_ref]
nint_ref = nint[nod_ref]

VN = (affinity-aff_ref) + vt*np.log(NC/NC_ref)
VP = -(affinity-aff_ref) - (egap - egap_ref) + vt*np.log(NV/NV_ref)

# PARAMETROS en los puntos medios entre nodos
eps_avg = (eps[:-1]+eps[1:])/2
movn_avg = (movn[:-1]+movn[1:])/2
movp_avg = (movp[:-1]+movp[1:])/2

# CONSTANTES de normalización (scaling)
v0 = vt
l0 = long
n0 = max([max(abs(c)), max(nint)])
dif0 = max([movn0_gaas, movp0_gaas])*v0  # [cm^2/seg]
t0 = np.power(l0, 2)/dif0
j0 = q*dif0*n0/l0
cdir0 = dif0/n0/np.power(l0, 2)  # Recombinacion directa
caug0 = cdir0/n0  # Recombinación Auger

# NORMALIZACION de constantes y arrays
nnint = nint/n0
ttn = tn/t0
ttp = tp/t0
nn_reg = n_reg/n0
pp_reg = p_reg/n0
cc = c/n0
epss_avg = eps_avg/eps_ref
movvn_avg = movn_avg*v0/dif0
movvp_avg = movp_avg*v0/dif0
vbii = vbi/v0
VVN = VN/v0
VVP = VP/v0
cc_dir_gaas = c_dir_gaas/cdir0
ccn_aug_gaas = cn_aug_gaas/caug0
ccp_aug_gaas = cp_aug_gaas/caug0
lambda_cuad = eps_ref*eps0*v0/(np.power(l0, 2)*q*n0)
hh1i = h1i/l0
hh2i = h2i/l0
cte_v = lambda_cuad/hh2i
cte_n = movvn_avg/hh1i
cte_p = movvp_avg/hh1i

# POLARIZACION 
# condiciones: pol>0 -> directa, pol<0 inversa, sin importar la disposición de los
# materiales n y p. La tensión pol se aplica en el último nodo de la malla. Si 
# pol>0 la altura de la barrera de potencial bajará, y si es <0 aumentará. En el
# primer nodo siempre la tensión es la de referencia, =0.
#pol = np.r_[-1.5:0:0.1, 0.1:1.6:0.1] # fig 7b Yang93
#pol = np.r_[-0.4:0.25:0.05]#, 0.05:0.25:0.05]
pol = np.array([0.])
elem_pol = len(pol)
jt = np.zeros(elem_pol)

# -----------------------------------------------------------------------------
# CALCULO de v, n y p en equilibrio térmico -----------------------------------
# -----------------------------------------------------------------------------

vv_act = np.zeros(celdas+1)
vv_ant = np.zeros(celdas+1)
nn_act = np.zeros(celdas+1)
nn_ant = np.zeros(celdas+1)
pp_act = np.zeros(celdas+1)
pp_ant = np.zeros(celdas+1)

# Soluciones iniciales para v, n y p
for i in range(1, num_reg):
    np.put(vv_act, np.r_[nod_interfaces2[i]:nod_interfaces2[i+1]], sum(vbii[0:i]))

for i in range(num_reg):
    np.put(nn_act, np.r_[nod_interfaces2[i]:nod_interfaces2[i+1]],nn_reg[i])
    np.put(pp_act, np.r_[nod_interfaces2[i]:nod_interfaces2[i+1]],pp_reg[i])

# Condiciones de contorno
vv0 = 0; vv1 = vv_act[-1]
nn0 = nn_reg[0]; nn1 = nn_reg[-1]
pp0 = pp_reg[0]; pp1 = pp_reg[-1]

k = 0
norma = 1

while norma > tol_equ and k < K_MAX:
    
    norma = np.linalg.norm(vv_act-vv_ant)
    print(norma)
    vv_ant = vv_act
    nn_ant = nn_act
    pp_ant = pp_act
    k += 1

    # Método de Gummel para cálculo de v, n y p -------------------------------
    
    # 1- CALCULO de v    
        
    # Matriz tri-diagonal
    mat_vv = np.diag(cte_v[1:]*(epss_avg[1:]/hh1i[1:]+epss_avg[:-1]/hh1i[:-1]) + \
    pp_ant[1:-1]+nn_ant[1:-1]) + \
    np.diag(-cte_v[1:-1]*epss_avg[1:-1]/hh1i[1:-1], 1) + \
    np.diag(-cte_v[2:]*epss_avg[1:-1]/hh1i[1:-1], -1)
    # Vector rhs
    vec_vv = pp_ant[1:-1]*(vv_ant[1:-1]+1)+nn_ant[1:-1]*(vv_ant[1:-1]-1)+cc[1:-1]
    vec_vv[0] = vec_vv[0]+cte_v[1]/hh1i[0]*vv_ant[0]
    vec_vv[-1] = vec_vv[-1]+cte_v[-1]*epss_avg[-1]/hh1i[-1]*vv_ant[-1]
    # Cálculo de v: se resuelve el sistema Ax=b
    vv_act = np.linalg.solve(mat_vv, vec_vv)
    vv_act = np.r_[vv0, vv_act, vv1]
    
    print(k)
    
    # 2- CALCULO de n    
    
    # TASA neta de recombinación-generación
    r_g = (nn_act*pp_act - nnint**2)/(ttp*(nn_act + nnint) + ttn*(pp_act + nnint))
    
    # Matriz tri-diagonal
    mat_nn = np.diag(cte_n[:-1]/hh2i[1:]*bernu(vv_act[1:-1]+VVN[1:-1]-(vv_act[:-2]+VVN[:-2])) + \
    cte_n[1:]/hh2i[1:]*bernu(vv_act[1:-1]+VVN[1:-1]-(vv_act[2:]+VVN[2:]))) + \
    np.diag(-cte_n[1:-1]/hh2i[1:-1]*bernu(vv_act[2:-1]+VVN[2:-1]-(vv_act[1:-2]+VVN[1:-2])), 1) + \
    np.diag(-cte_n[1:-1]/hh2i[2:]*bernu(vv_act[1:-2]+VVN[1:-2]-(vv_act[2:-1]+VVN[2:-1])), -1)
    # Vector rhs
    vec_nn = r_g[1:-1]
    vec_nn[0] = vec_nn[0] + cte_n[0]/hh2i[1]*bernu(vv_act[0]+VVN[0]-(vv_act[1]+VVN[1]))*nn_ant[0]
    vec_nn[-1] = vec_nn[-1] + cte_n[-1]/hh2i[-1]*bernu(vv_act[-1]+VVN[-1]-(vv_act[-2]+VVN[-2]))*nn_ant[-1]
    # Cálculo de n: se resuelve el sistema Ax=b
    nn_act = np.linalg.solve(mat_nn, vec_nn)
    nn_act = np.r_[nn0, nn_act, nn1]
    
    # 3- CALCULO de p
    
    # TASA neta de recombinación-generación
    r_g = (nn_act*pp_act - nnint**2)/(ttp*(nn_act + nnint) + ttn*(pp_act + nnint))
    
    # Matriz tri-diagonal
    mat_pp = np.diag(cte_p[:-1]/hh2i[1:]*bernu(vv_act[:-2]-VVP[:-2]-(vv_act[1:-1]-VVP[1:-1])) + \
    cte_p[1:]/hh2i[1:]*bernu(vv_act[2:]-VVP[2:]-(vv_act[1:-1]-VVP[1:-1]))) + \
    np.diag(-cte_p[1:-1]/hh2i[1:-1]*bernu(vv_act[1:-2]-VVP[1:-2]-(vv_act[2:-1]-VVP[2:-1])), 1) + \
    np.diag(-cte_p[1:-1]/hh2i[2:]*bernu(vv_act[2:-1]-VVP[2:-1]-(vv_act[1:-2]-VVP[1:-2])), -1)
    # Vector rhs
    vec_pp = r_g[1:-1]
    vec_pp[0] = vec_pp[0] + cte_p[0]/hh2i[1]*bernu(vv_act[1]-VVP[1]-(vv_act[0]-VVP[0]))*pp_ant[0]
    vec_pp[-1] = vec_pp[-1] + cte_p[-1]/hh2i[-1]*bernu(vv_act[-2]-VVP[-2]-(vv_act[-1]-VVP[-1]))*pp_ant[-1]
    # Cálculo de p: se resuelve el sistema Ax=b
    pp_act = np.linalg.solve(mat_pp, vec_pp)
    pp_act = np.r_[pp0, pp_act, pp1]
    
# FIN del while solución en equilibrio térmico
    
if k==K_MAX:
    print('Problemas de convergencia al calcular solución en equilibrio')

# -----------------------------------------------------------------------------
# CALCULO de v, n y p bajo polarización fuera de equilibrio -------------------
# -----------------------------------------------------------------------------

vva = pol/v0
vva_ant = 0

for j in range(elem_pol):
    
    # Estimación inicial de v en función del potencial anterior
    #vv_act = vv_act*(vv_act[-1]-(vva[j]-vva_ant))/vv_act[-1] (ANTERIOR-MAL)
    # pol>0 representa una polarización en directa -> la barrera de potencial se 
    # achica. pol<0 representa polarización inversa -> la barrera aumenta.
    # Para encontrar la estimación inicial de v y fijar la condición de contorno
    # vv1, tengo en cuenta el signo de vv_act en el último nodo, donde se aplica
    # la polarización en cuestión. Si vv_act[-1]<0 a vv_act[-1] le sumo pol, y 
    # si es >=0 se lo resto.
    vv_act = vv_act*(vv_act[-1] - (vv_act[-1]>=0)*(vva[j]-vva_ant) + \
    (vv_act[-1]<0)*(vva[j]-vva_ant))/vv_act[-1]
    vva_ant = vva[j]
    vv1 = vv_act[-1] # FIJO ASI LA CONDICION DE CONTORNO PARA EL NUEVO POTENCIAL
    
    k = 0
    normav = 1

    while normav > tol and k < K_MAX:
        
        normav = np.linalg.norm(vv_act-vv_ant)
        print(normav)
        norman = np.linalg.norm(nn_act-nn_ant)
        print(norman)
        normap = np.linalg.norm(pp_act-pp_ant)
        print(normap)
        vv_ant = vv_act
        nn_ant = nn_act
        pp_ant = pp_act
        k += 1
        
        # Método de Gummel para cálculo de v, n y p -------------------------------
    
        # 1- CALCULO de v    
        
        # Matriz tri-diagonal
        mat_vv = np.diag(cte_v[1:]*(epss_avg[1:]/hh1i[1:]+epss_avg[:-1]/hh1i[:-1]) + \
        pp_ant[1:-1]+nn_ant[1:-1]) + \
        np.diag(-cte_v[1:-1]*epss_avg[1:-1]/hh1i[1:-1], 1) + \
        np.diag(-cte_v[2:]*epss_avg[1:-1]/hh1i[1:-1], -1)
        # Vector rhs
        vec_vv = pp_ant[1:-1]*(vv_ant[1:-1]+1)+nn_ant[1:-1]*(vv_ant[1:-1]-1)+cc[1:-1]
        vec_vv[0] = vec_vv[0]+cte_v[1]/hh1i[0]*vv_ant[0]
        vec_vv[-1] = vec_vv[-1]+cte_v[-1]*epss_avg[-1]/hh1i[-1]*vv_ant[-1]
        # Cálculo de v: se resuelve el sistema Ax=b
        vv_act = np.linalg.solve(mat_vv, vec_vv)
        vv_act = np.r_[vv0, vv_act, vv1]
    
        print(k)
    
        # 2- CALCULO de n    
    
        # TASA neta de recombinación-generación
        r_g = (nn_act*pp_act - nnint**2)/(ttp*(nn_act + nnint) + ttn*(pp_act + nnint))
    
        # Matriz tri-diagonal
        mat_nn = np.diag(cte_n[:-1]/hh2i[1:]*bernu(vv_act[1:-1]+VVN[1:-1]-(vv_act[:-2]+VVN[:-2])) + \
        cte_n[1:]/hh2i[1:]*bernu(vv_act[1:-1]+VVN[1:-1]-(vv_act[2:]+VVN[2:]))) + \
        np.diag(-cte_n[1:-1]/hh2i[1:-1]*bernu(vv_act[2:-1]+VVN[2:-1]-(vv_act[1:-2]+VVN[1:-2])), 1) + \
        np.diag(-cte_n[1:-1]/hh2i[2:]*bernu(vv_act[1:-2]+VVN[1:-2]-(vv_act[2:-1]+VVN[2:-1])), -1)
        # Vector rhs
        vec_nn = r_g[1:-1]
        vec_nn[0] = vec_nn[0] + cte_n[0]/hh2i[1]*bernu(vv_act[0]+VVN[0]-(vv_act[1]+VVN[1]))*nn_ant[0]
        vec_nn[-1] = vec_nn[-1] + cte_n[-1]/hh2i[-1]*bernu(vv_act[-1]+VVN[-1]-(vv_act[-2]+VVN[-2]))*nn_ant[-1]
        # Cálculo de n: se resuelve el sistema Ax=b
        nn_act = np.linalg.solve(mat_nn, vec_nn)
        nn_act = np.r_[nn0, nn_act, nn1]
    
        # 3- CALCULO de p
    
        # TASA neta de recombinación-generación
        r_g = (nn_act*pp_act - nnint**2)/(ttp*(nn_act + nnint) + ttn*(pp_act + nnint))
    
        # Matriz tri-diagonal
        mat_pp = np.diag(cte_p[:-1]/hh2i[1:]*bernu(vv_act[:-2]-VVP[:-2]-(vv_act[1:-1]-VVP[1:-1])) + \
        cte_p[1:]/hh2i[1:]*bernu(vv_act[2:]-VVP[2:]-(vv_act[1:-1]-VVP[1:-1]))) + \
        np.diag(-cte_p[1:-1]/hh2i[1:-1]*bernu(vv_act[1:-2]-VVP[1:-2]-(vv_act[2:-1]-VVP[2:-1])), 1) + \
        np.diag(-cte_p[1:-1]/hh2i[2:]*bernu(vv_act[2:-1]-VVP[2:-1]-(vv_act[1:-2]-VVP[1:-2])), -1)
        # Vector rhs
        vec_pp = r_g[1:-1]
        vec_pp[0] = vec_pp[0] + cte_p[0]/hh2i[1]*bernu(vv_act[1]-VVP[1]-(vv_act[0]-VVP[0]))*pp_ant[0]
        vec_pp[-1] = vec_pp[-1] + cte_p[-1]/hh2i[-1]*bernu(vv_act[-2]-VVP[-2]-(vv_act[-1]-VVP[-1]))*pp_ant[-1]
        # Cálculo de p: se resuelve el sistema Ax=b
        pp_act = np.linalg.solve(mat_pp, vec_pp)
        pp_act = np.r_[pp0, pp_act, pp1]
        
        # CORRIENTES
        #jn = j0*cte_n*(nn_act[:-1]*bernu(vv_act[:-1]+VVN[:-1]-(vv_act[1:]+VVN[1:])) - \
        #nn_act[1:]*bernu(vv_act[1:]+VVN[1:]-(vv_act[:-1]+VVN[:-1])))

        #jp = j0*cte_p*(-pp_act[:-1]*bernu(vv_act[1:]-VVP[1:]-(vv_act[:-1]-VVP[:-1])) + \
        #pp_act[1:]*bernu(vv_act[:-1]-VVP[:-1]-(vv_act[1:]-VVP[1:])))
    
        #jt[j] = jn[centro] + jp[centro]
        #print(jt[j])
    
    # FIN del while solución fuera del equilibrio térmico
        
    # CORRIENTES
    jn = j0*cte_n*(nn_act[:-1]*bernu(vv_act[:-1]+VVN[:-1]-(vv_act[1:]+VVN[1:])) - \
    nn_act[1:]*bernu(vv_act[1:]+VVN[1:]-(vv_act[:-1]+VVN[:-1])))

    jp = j0*cte_p*(-pp_act[:-1]*bernu(vv_act[1:]-VVP[1:]-(vv_act[:-1]-VVP[:-1])) + \
    pp_act[1:]*bernu(vv_act[:-1]-VVP[:-1]-(vv_act[1:]-VVP[1:])))
    
    jt[j] = jn[centro] + jp[centro]
    
    if k==K_MAX:
        print('Problemas de convergencia al calcular solución fuera del equilibrio')
        break


print(jt)

# CAMPO eléctrico
e = campo_electrico(vv_act*v0, h1i)

# BANDAS de energía
V0 = aff_ref + vt*np.log(NC_ref/nint_ref)
EC = V0 - vv_act*v0 - affinity
EV = EC - egap

# ARCHIVO txt con resultados
t = datetime.datetime.now()
fechayhora = t.strftime('%d%b%Y_%H_%M_%S')
np.savetxt("hetero4_"+fechayhora+".dat", np.transpose((x, EC, EV, vv_act*v0, nn_act*n0, pp_act*n0)))


# GRAFICOS de v, e, n y p

pyplot.figure()
pyplot.plot(x*1e7, EC, 'b', label='Banda de conducción')
pyplot.plot(x*1e7, EV, 'g', label='Banda de valencia')
#pyplot.axis([0, 1000, -1.0, 2.5])
pyplot.title('Diagrama de bandas de energía',fontsize=20)
pyplot.xlabel('x [nm]',fontsize=18)
pyplot.ylabel('Energía [eV]',fontsize=18)
pyplot.tick_params(axis='both', labelsize=18)
pyplot.legend(loc=3)
pyplot.grid(True)
pyplot.savefig("bandas_"+fechayhora+".png", dpi=300)

pyplot.figure()
pyplot.plot(x*1e7, vv_act*v0)
pyplot.title('Potencial',fontsize=18)
pyplot.xlabel('x [nm]',fontsize=18)
pyplot.ylabel('Potencial [V]',fontsize=18)
pyplot.tick_params(axis='both', labelsize=18)
pyplot.grid(True)
pyplot.savefig("potencial_"+fechayhora+".png", dpi=300)

pyplot.figure()
pyplot.semilogy(x*1e7, nn_act*n0,'b', label='Electrones')
pyplot.semilogy(x*1e7, pp_act*n0,'r', label='Huecos')
pyplot.title('Concentración de portadores',fontsize=18)
pyplot.xlabel('x [nm]',fontsize=18)
pyplot.ylabel('Concentración [cm-3]',fontsize=18)
pyplot.tick_params(axis='both', labelsize=18)
pyplot.legend(loc=4)
pyplot.grid(True)
pyplot.savefig("electrones_huecos_"+fechayhora+".png", dpi=300)

pyplot.figure()
pyplot.plot(x[1:]*1e7, e/1e3)
pyplot.title('Campo eléctrico',fontsize=18)
pyplot.xlabel('x [nm]',fontsize=18)
pyplot.ylabel('Campo eléctrico [kV/cm]',fontsize=18)
pyplot.tick_params(axis='both', labelsize=18)
pyplot.grid(True)
pyplot.savefig("campo_"+fechayhora+".png", dpi=300)

pyplot.figure()
pyplot.plot(x[1:]*1e7, jn, 'b', label='Densidad de corriente de electrones')
pyplot.plot(x[1:]*1e7, jp, 'r', label='Densidad de corriente de huecos')
pyplot.plot(x[1:]*1e7, jn+jp, 'k', label='Densidad de corriente total')
pyplot.xlabel('x [nm]',fontsize=18)
pyplot.ylabel('Densidad de corriente [A/cm2]',fontsize=18)
pyplot.tick_params(axis='both', labelsize=18)
pyplot.legend(loc=7)

if elem_pol>1:
    pyplot.figure(figsize=(5,6))
    pyplot.semilogy(pol, abs(jt), 'o')
    #pyplot.axis([-1.5, 1.5, 1e-11, 1e5])
    pyplot.xlabel('Tensión [V]',fontsize=20)
    pyplot.ylabel('Densidad de corriente [A/cm2]',fontsize=20)
    pyplot.tick_params(axis='both', labelsize=20)
    pyplot.axvline(color='k')
    pyplot.savefig("jv_"+fechayhora+".png", dpi=300)
    np.savetxt("jv_"+fechayhora+".dat", np.transpose((pol, jt)))