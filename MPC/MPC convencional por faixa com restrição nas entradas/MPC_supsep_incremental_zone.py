#from mpcconvencional import *
from icmpc_zone import *
import numpy as np
import time
import seaborn as sns

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
q = [1, 1, 1]
r = [1/(3000**2), 1/(100**2), 1/(100**2)]
    
# Restrições
dumax = [60, 1.5, 1.5]
umax = [7200, 100, 100]
umin = [4200, 0, 0]
# ymax = [1.78e4, 400.15, 393.15]
# ymin = [1.43e4, 280, 300]

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
nsim = 200
ur, yk, tcalc, Vk, success_vals, y_sp, Ymax, Ymin = [], [], [], [], [], [], [], []
for k in range(nsim):

    # ymin = [1.70e4, 319.52, 341.37]
    # ymax = [1.70e4, 319.52, 341.37]
    if k <= 100:
        ymin = [1.70e4, 319.15, 250]
        ymax = [1.72e4, 325, 350]
    elif k > 100 and k < 200:
        ymin = [1.72e4, 319.15, 250]
        ymax = [1.73e4, 325, 350]

    start_time = time.time()
    solutionit = MPCIncremental(m, p, q, r, A, B, C, nx, nu, ny, 
                                 ymax-yss.flatten(), ymin-yss.flatten(), dumax, 
                                 umax-uss.flatten(), umin-uss.flatten(), 
                                 xmk, uk_1)
    solution = solutionit.control()

    tcalc.append(time.time() - start_time)

    duk, ysp, vk, status_code = solution
    xmk = A@xmk + B@duk
    ymk = C@xmk
    # xpk = Ap@xpk + Bp@duk
    # ypk = Cp@xpk
    xpk = xmk
    ypk = ymk
    uk_1 = duk + uk_1
    # de = ypk - ymk
    # xmk = xmk + Kf@de
    
    # Gravar valores absolutos para plotagem
    ur.append((uk_1 + uss).flatten())
    yk.append((ypk + yss).flatten())
    y_sp.append((ysp + yss).flatten())
    Ymax.append((np.array(ymax)).flatten())
    Ymin.append((np.array(ymin)).flatten())
    Vk.append(vk)
    success_vals.append(status_code)

# Reshape para plotar
ur = np.array(ur)
yk = np.array(yk)
y_sp = np.array(y_sp)
Vk = np.array(Vk).reshape(-1,1)
Ymax = np.array(Ymax)
Ymin = np.array(Ymin)
tcalc = np.array(tcalc)
success_vals = np.array(success_vals)

# Criar vetores de limites para plotagem (constantes)
Umax = np.tile(umax, (nsim, 1))
Umin = np.tile(umin, (nsim, 1))
# Ymax = np.tile(ymax, (nsim, 1))
# Ymin = np.tile(ymin, (nsim, 1))

plot = Plot(fontsize=14)
resultado = plot.criar_pasta_simulacao()
line = 2

# =============== Subplot com variáveis de controle ===============
plot.plotar(
    x = np.arange(nsim),
    y_list = [
        [ur[:,0], Umin[:,0], Umax[:,0]],
        [ur[:,1], Umin[:,1], Umax[:,1]],
        [ur[:,2], Umin[:,2], Umax[:,2]]
    ],
    labels = [
        ['$u(k)$', '$u_{min,max}$', ''],
        ['$u(k)$', '$u_{min,max}$', ''],
        ['$u(k)$', '$u_{min,max}$', '']
    ],
    xlabel = 'Time /(min)',
    ylabel = [r'$u_1~/(rpm)$', r'$u_2~/(\%)$', r'$u_3~/(\%)$'],
    cores = [
        ['black', 'red', 'red'],
        ['black', 'red', 'red'],
        ['black', 'red', 'red']
    ],
    estilos = [
        ['-', '-.', '-.'],
        ['-', '-.', '-.'],
        ['-', '-.', '-.']
    ],
    line = [
        [line, line, line],
        [line, line, line],
        [line, line, line]
    ],
    legenda_ncol = 3,
    subplots = 'vertical',
    salvar = 'control_variables.png'
)

# =============== Compressor speed ===============
plot.plotar(
    x = np.arange(len(ur[:,0])),
    y_list= [ur[:,0],Umin[:,0],Umax[:,0]],
    labels=['$u(k)$','$u_{min,max}$',''],
    xlabel='Time /(min)',
    ylabel=r'$u_1~/(rpm)$',
    cores = ['black', 'red', 'red'],
    marcadores = ['','', ''],
    estilos=['-','-.','-.'],
    line = line,
    salvar = 'u1_pump_speed.png'
)

# =============== Valve bypass 1 ===============
plot.plotar(
    x = np.arange(len(ur[:,1])),
    y_list = [ur[:,1], Umin[:,1], Umax[:,1]],
    labels=['$u(k)$','$u_{min,max}$',''],
    xlabel='Time /(min)',
    ylabel=r'$u_2~/(\%)$',
    cores = ['black','red','red'],
    marcadores = ['','', ''],
    estilos=['-','-.','-.'],
    line = line,
    salvar = 'u2_valve_bypass_1.png'
)

# =============== Valve bypass 2 ===============
plot.plotar(
    x = np.arange(len(ur[:,2])),
    y_list = [ur[:,2], Umin[:,2], Umax[:,2]],
    labels=['$u(k)$','$u_{min,max}$',''],
    xlabel='Time /(min)',
    ylabel=r'$u_3~/(\%)$',
    cores = ['black','red','red'],
    marcadores = ['','', ''],
    estilos=['-','-.','-.'],
    line = line,
    salvar = 'u3_valve_bypass_2.png'
)

# =============== Subplot com variáveis de saída ===============
plot.plotar(
    x = np.arange(nsim),
    y_list = [
        [yk[:,0], y_sp[:,0], Ymax[:,0], Ymin[:,0]],
        [yk[:,1], y_sp[:,1], Ymax[:,1], Ymin[:,1]],
        [yk[:,2], y_sp[:,2], Ymax[:,2], Ymin[:,2]]
    ],
    labels = [
        ['$y(k)$', '$y_{sp}$', '$y_{min,max}$', ''],
        ['$y(k)$', '$y_{sp}$', '$y_{min,max}$', ''],
        ['$y(k)$', '$y_{sp}$', '$y_{min,max}$', '']
    ],
    xlabel = 'Time /(min)',
    ylabel = [r'$y_1~/(bar)$', r'$y_2~/(^\circ C)$', r'$y_3~/(^\circ C)$'],
    cores = [
        ['black', 'green', 'red', 'red'],
        ['black', 'green', 'red', 'red'],
        ['black', 'green', 'red', 'red']
    ],
    estilos = [
        ['-', '--', '-.', '-.'],
        ['-', '--', '-.', '-.'],
        ['-', '--', '-.', '-.']
    ],
    line = [
        [line, line, line, line],
        [line, line, line, line],
        [line, line, line, line]
    ],
    legenda_ncol = 4,
    subplots = 'vertical',
    salvar = 'output_variables.png'
)

# =============== Pressure Scrubber ===============
plot.plotar(
    x = np.arange(len(yk[:,0])),
    y_list = [yk[:,0],y_sp[:,0], Ymax[:,0], Ymin[:,0]],
    labels=['$y(k)$','$y_{sp}$','$y_{min,max}$',''],
    xlabel='Time /(min)',
    ylabel=r'$y_1~/(bar)$',
    cores = ['black', 'green', 'red', 'red'],
    marcadores = ['', '', '', ''],
    estilos=['-', '--', '-.', '-.'],
    line = line,
    salvar = 'y1_pressure_scrubber.png'
)

# =============== Temperature in scrubber in ===============
plot.plotar(
    x = np.arange(len(yk[:,1])),
    y_list = [yk[:,1],y_sp[:,1], Ymax[:,1], Ymin[:,1]],
    labels=['$y(k)$','$y_{sp}$','$y_{min,max}$',''],
    xlabel='Time /(min)',
    ylabel=r'$y_2~/(^\circ C)$',
    cores = ['black','green','red', 'red'],
    marcadores = ['','', '', ''],
    estilos=['-','--','-.', '-.'],
    line = line,
    salvar = 'y2_temperature_scrubber_in.png'
)

# =============== Temperature in compressor out ===============
plot.plotar(
    x = np.arange(len(yk[:,2])),
    y_list = [yk[:,2],y_sp[:,2], Ymax[:,2], Ymin[:,2]],
    labels=['$y(k)$','$y_{sp}$','$y_{min,max}$',''],
    xlabel='Time /(min)',
    ylabel=r'$y_3~/(^\circ C)$',
    cores = ['black','green','red', 'red'],
    marcadores = ['','', '', ''],
    estilos=['-','--','-.', '-.'],
    line = line,
    salvar = 'y3_temperature_reinjection.png'
)

# ========== Execution Time Histogram ===========
fig_time = plt.figure()
tcalc_ms = tcalc * 1000  # Converter para milissegundos
media_tcalc_ms = np.mean(tcalc_ms)

sns.histplot(tcalc_ms, bins=30, kde=True, color='red', edgecolor='black', alpha=0.6, line_kws={'color': 'blue', 'linewidth': line})
plt.axvline(media_tcalc_ms, color='green', linestyle='--', linewidth=line, label=f'Average: {media_tcalc_ms:.2f} ms')
plt.xlabel('Execution Time /(ms)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.legend(fontsize=14, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(resultado, 'execution_time.png'), dpi=600, bbox_inches='tight')

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
    y_list=[success_vals],
    labels=[''],
    xlabel='Time /(min)',
    ylabel='Status',
    cores = ['black'],
    marcadores = [''],
    estilos = ['-'],
    line = line,
    salvar = 'status_optimization.png'
)
plt.show()