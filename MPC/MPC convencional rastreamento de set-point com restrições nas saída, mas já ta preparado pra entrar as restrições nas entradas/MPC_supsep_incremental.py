#from mpcconvencional import *
from icmpc import *
import numpy as np
import time

# Modelo em espaço de estados
Atil = np.loadtxt('./Atil.txt')
Btil = np.loadtxt('./Btil.txt')
Ctil = np.loadtxt('./Ctil.txt')

inc = Incremental(Atil, Btil, Ctil)
A, B, C = inc.A, inc.B, inc.C

# Modelo da planta
Ap = A
Bp = B
Cp = C

# Dimensões das variáveis
nx = A.shape[0] # Número de estados
nu = B.shape[1] # Número de entradas
ny = C.shape[0] # Número de saídas

# Parâmetros do controlador
m = 3
p = 70
q = [1/(0.35e4**2), 1/(120**2), 10/(93.15**2)]
r = [1/(3000**2), 1/(100**2), 10/(100**2)]
    
# Restrições
dumax = [60, 0.8, 0.8]
umax = [7200, 100, 100]
umin = [4200, 0, 0]
ymax = [1.78e4, 400.15, 393.15]
ymin = [1.43e4, 280, 300]

# Estado estacionário
uss = np.array([[4800], [50], [10]])
yss = np.array([[1.70e4], [319.52], [341.37]])

# Condições iniciais (variáveis de desvio)
xmk = np.zeros((nx,1))
xpk = xmk
ymk = C@xmk
ypk = Cp@xpk
uk_1 = uss - uss  # = zeros (desvios no equilíbrio)

def f_kalman(ny, A, C, it):
    V = 0.5
    W = 0.5
    sM = A.shape[0]
    PP = np.eye(sM)
    VV = np.eye(ny) * V
    WW = np.eye(sM) * W

    for _ in range(it):
        PP = A @ PP @ A.T - A @ PP @ C.T @ np.linalg.inv(VV + C @ PP @ C.T) @ C @ PP @ A.T + WW

    Kalman = A @ PP @ C.T @ np.linalg.inv(VV + C @ PP @ C.T)
    return Kalman

Kf = f_kalman(ny,A,C,100)

# Iniciando simulação
nsim = 300
ur, yk, tcalc, Vk, success_vals, y_sp = [], [], [], [], [], []
for k in range(nsim):
    ys = np.array([[1.70e4], [319.52], [341.37]])

    if k <= 100:
        ys = np.array([[1.70e4], [319.52], [341.37]])
    elif k > 100 and k < 200:
        ys = np.array([[1.68e4], [320.52], [342]])
    elif k >= 200 and k < 300:
        ys = np.array([[1.70e4], [319.52], [341.37]])

    start_time = time.time()
    solutionit = MPCIncremental(m, p, q, r, A, B, C, nx, nu, ny, 
                                 ymax-yss.flatten(), ymin-yss.flatten(), dumax, 
                                 umax-uss.flatten(), umin-uss.flatten(), 
                                 xmk, uk_1, (ys-yss).flatten().tolist())
    solution = solutionit.control()

    tcalc.append(time.time() - start_time)

    duk, vk, status_code = solution
    xmk = A@xmk + B@duk
    ymk = C@xmk
    xpk = Ap@xpk + Bp@duk
    ypk = Cp@xpk

    uk_1 = duk + uk_1
    # de = ypk - ymk
    # xmk = xmk + Kf@de
    
    # Gravar valores absolutos para plotagem
    ur.append((uk_1 + uss).flatten())
    yk.append((ypk + yss).flatten())
    y_sp.append(ys.flatten())
    Vk.append(vk)
    success_vals.append(status_code)

# Reshape para plotar
ur = np.array(ur)
yk = np.array(yk)
y_sp = np.array(y_sp)
Vk = np.array(Vk).reshape(-1,1)
tcalc = np.array(tcalc)
success_vals = np.array(success_vals)

# Criar vetores de limites para plotagem (constantes)
Umax = np.tile(umax, (nsim, 1))
Umin = np.tile(umin, (nsim, 1))
Ymax = np.tile(ymax, (nsim, 1))
Ymin = np.tile(ymin, (nsim, 1))

plot = Plot(fontsize=14)
resultado = plot.criar_pasta_simulacao()
line = 2
# =============== Compressor speed ===============
plot.plotar(
    x = np.arange(len(ur[:,0])),
    y_list= [ur[:,0],Umin[:,0],Umax[:,0]],
    labels=['uk','uk_{min,max}',''],
    xlabel='Time /(min)',
    ylabel='Speed /(rpm)',
    cores = ['black', 'red', 'red'],
    marcadores = ['','', ''],
    estilos=['-','-.','-.'],
    line = line,
    salvar = 'compressor_speed_control.png'
)

# =============== Valve bypass 1 ===============
plot.plotar(
    x = np.arange(len(ur[:,1])),
    y_list = [ur[:,1], Umin[:,1], Umax[:,1]],
    labels=['uk','uk_{min,max}',''],
    xlabel='Time /(min)',
    ylabel='Opening /(%)',
    cores = ['black','red','red'],
    marcadores = ['','', ''],
    estilos=['-','-.','-.'],
    line = line,
    salvar = 'valve_bypass_1_control.png'
)

# =============== Valve bypass 2 ===============
plot.plotar(
    x = np.arange(len(ur[:,2])),
    y_list = [ur[:,2], Umin[:,2], Umax[:,2]],
    labels=['uk','uk_{min,max}',''],
    xlabel='Time /(min)',
    ylabel='Opening /(%)',
    cores = ['black','red','red'],
    marcadores = ['','', ''],
    estilos=['-','-.','-.'],
    line = line,
    salvar = 'valve_bypass_2_control.png'
)

# =============== Pressure Scrubber ===============
plot.plotar(
    x = np.arange(len(yk[:,0])),
    y_list = [yk[:,0],y_sp[:,0], Ymax[:,0], Ymin[:,0]],
    labels=['yk','y_{sp}','y_{min,max}',''],
    xlabel='Time /(min)',
    ylabel='Pressure /(bar)',
    cores = ['black', 'green', 'red', 'red'],
    marcadores = ['', '', '', ''],
    estilos=['-', '--', '-.', '-.'],
    line = line,
    salvar = 'pressure_scrubber.png'
)

# =============== Temperature in scrubber in ===============
plot.plotar(
    x = np.arange(len(yk[:,1])),
    y_list = [yk[:,1],Ymin[:,1],Ymax[:,1],y_sp[:,1]],
    labels=['yk','yk_{min,max}','','y_{sp}'],
    xlabel='Time /(min)',
    ylabel='Temperature /(°C)',
    cores = ['black','red','red', 'green'],
    marcadores = ['','', '', ''],
    estilos=['-','-.','-.', '--'],
    salvar = 'temperature_scrubber_in.png'
)

# =============== Temperature in compressor out ===============
plot.plotar(
    x = np.arange(len(yk[:,2])),
    y_list = [yk[:,2],Ymin[:,2],Ymax[:,2],y_sp[:,2]],
    labels=['yk','yk_{min,max}','','y_{sp}'],
    xlabel='Time /(min)',
    ylabel='Temperature /(°C)',
    cores = ['black','red','red', 'green'],
    marcadores = ['','', '', ''],
    estilos=['-','-.','-.', '--'],
    salvar = 'temperature_scrubber_out.png'
)

# ========== Cost Function ===========
plot.plotar(
    x=np.arange(len(Vk)),
    y_list=[Vk],
    labels=[''],
    xlabel='Time /(min)',
    ylabel='Cost function',
    cores=['black'],
    marcadores=[''],
    estilos=['-'],
    line = line,
    salvar = 'cost_function.png'
)

# =========== Status optimization ===========
plot.plotar(
    x = np.arange(len(success_vals)),
    y_list=[success_vals],         # Coloque dentro de uma lista
    labels=[''],                   # labels como lista
    xlabel='Time /(min)',
    ylabel='Status',
    cores = ['black'],
    marcadores = [''],
    estilos = ['-'],
    line = line,
    salvar = 'status_optimization.png'
)
plt.show()