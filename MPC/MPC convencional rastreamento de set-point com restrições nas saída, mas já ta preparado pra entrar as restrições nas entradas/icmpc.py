"""
Autor: Fernando Calhau
GitHub: https://github.com/fernandocalhau
LinkedIn: https://www.linkedin.com/in/fernando-calhau-740816141/
Lattes: https://lattes.cnpq.br/9464620638306106
E-mail: fernando.s.calhau@gmail.com
"""

import numpy as np
from numpy.linalg import multi_dot, matrix_power, inv, eig, norm, cond
from cvxopt import matrix, solvers
import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error, r2_score
import sympy as sp
logging.basicConfig(level=logging.INFO)

class Incremental:
    def __init__(self, A, B, C):
        # number of quantities
        self.nu = B.shape[1]
        self.nx_system = A.shape[0]
        self.ny = C.shape[0]

        # Matrices into the velocity form (incremental model)
        A_ = sp.Matrix(np.vstack((np.hstack((A,B)),np.hstack((np.zeros((self.nu,self.nx_system)),np.eye(self.nu,self.nu))))))
        B_ = np.vstack((B,np.eye(self.nu,self.nu)))
        C_ = np.hstack((C,np.zeros((self.ny,self.nu))))

        self.A = np.array(A_,dtype=float)
        self.B = B_
        self.C = C_

class Kalman():

    def __init__(self,A, C, it):

        ny = C.shape[0]
        V = .0001 # planta
        #W = .0001 # modelo
        W = 0.002 # modelo  modificado
        sM = A.shape[0]
        Pn_1 = np.eye(sM,sM)
        VV = np.dot(np.eye(ny,ny), V)
        WW = np.dot(np.eye(sM,sM), W)
        for j in range(0,it):

            Rn = multi_dot([A, Pn_1, A]).transpose()+WW
            Pn = Rn - multi_dot([Rn, C.transpose(), inv(VV+multi_dot([C, Rn, C.transpose()])), C, Rn])
            Pn_1 = Pn

        self.gain = multi_dot([Rn, C.transpose(), inv(VV+ multi_dot([C, Rn, C.transpose()]))])


class MPCIncremental:
    def __init__(self, m, p, q, r, A, B, C, nx, nu, ny, ymax, ymin, dumax, umax, umin, xmk, uk_1, ysp):
        self.m = m
        self.p = p
        self.q = q
        self.r = r
        self.A = A
        self.B = B
        self.C = C
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.ymax = ymax
        self.ymin = ymin
        self.dumax = dumax
        self.umax = umax
        self.umin = umin
        self.xmk = xmk
        self.uk_1 = uk_1
        self.ysp = ysp

    def control(self):
        dumax = np.array(self.dumax, ndmin=2, dtype=float).transpose()
        umin  = np.array(self.umin, ndmin=2, dtype=float).transpose()
        umax  = np.array(self.umax, ndmin=2, dtype=float).transpose()
        ymin  = np.array(self.ymin, ndmin=2, dtype=float).transpose()
        ymax  = np.array(self.ymax, ndmin=2, dtype=float).transpose()
        ysp   = np.array(self.ysp, ndmin=2, dtype=float).transpose()


        q = np.array(self.q, ndmin=2, dtype=float)
        r = np.array(self.r, ndmin=2, dtype=float)

        # Phi: predição livre (matriz de observabilidade estendida)
        Phi = np.vstack([np.dot(self.C, np.linalg.matrix_power(self.A, k)) for k in range(1, self.p+1)])
        # Theta: resposta ao impulso (matriz de controle preditivo)
        tha = np.vstack([multi_dot([self.C, matrix_power(self.A, k-1), self.B]) for k in range(1, self.p+1)])
        
        aux1 = np.tile(q, (self.p, 1))
        aux2 = np.tile(r, (self.m, 1))
        

        a = tha.copy()
        Theta = a.copy()
        for iu in range(self.m-1):
            a = np.vstack([np.zeros((self.ny, self.nu)), a[0:(self.p-1)*self.ny, :]])
            Theta = np.hstack([Theta, a])

        Qbar = np.diag(aux1.flatten())
        Rbar = np.diag(aux2.flatten())

        Mtil  = np.kron(np.tril(np.ones((self.m, self.m))), np.eye(self.nu))
        Ymax  = np.tile(ymax, (self.p, 1))
        Ymin  = np.tile(ymin, (self.p, 1))
        Itil  = np.tile(np.eye(self.nu), (self.m, 1))
        Umax  = np.tile(umax, (self.m, 1))
        Umin  = np.tile(umin, (self.m, 1))
        Dumax = np.tile(dumax, (self.m, 1))
        Ysp   = np.tile(ysp, (self.p, 1))

        H = Theta.T@Qbar@Theta+Rbar
        H = (H + H.T)/2
        cf = ((Phi@self.xmk - Ysp).T @ Qbar @ Theta).T
        c  = (Phi@self.xmk - Ysp).T @ Qbar @ (Phi@self.xmk - Ysp)

        # Restrições de desigualdade Ain*x <= Bin
        Ain = np.vstack((Mtil,
                        -Mtil,
                        np.eye(self.m * self.nu),   
                        -np.eye(self.m * self.nu)))
        Bin = np.vstack([Umax - Itil @self.uk_1,
                         Itil @self.uk_1 - Umin,
                         Dumax,
                         Dumax])
            
        qp_opt = {'show_progress': False}
        solution = solvers.qp(matrix(H), matrix(cf), matrix(Ain), matrix(Bin), options=qp_opt)
        x = np.array(solution['x'])
        duk = x[0:self.nu]
        vk = x.T @ H @ x + 2 * cf.T @ x + c

        if solution['status'] == 'optimal':
            status_code = 1   # Problema viável
        else:
            status_code = -2  # Problema inviável

        return duk, vk, status_code


class Plot:
    def __init__(self, fontsize=14):
        """
        Inicializa a classe de plotagem com configurações padrão.

        Args:
            fontsize (int): Tamanho da fonte para os gráficos.
        """
        self.fontsize = fontsize
        self.pasta_base = None  # Pasta onde os gráficos serão salvos

    # Função para configurar os eixos de gráficos
    def config_plot(axes):
        """
        Configures the axes of Figures
        """
        formatter = ScalarFormatter(useOffset=False, useMathText=True)
        formatter.set_scientific(False)
        formatter.set_powerlimits((-1, 1))
        axes.yaxis.set_major_formatter(formatter)

        axes.xaxis.set_minor_locator(AutoMinorLocator())
        axes.yaxis.set_minor_locator(AutoMinorLocator())
        axes.tick_params(which='both', direction='out', bottom=True, left=True)
        axes.tick_params(which='major', width=2)
        axes.tick_params(which='minor', width=1)
        axes.xaxis.set_tick_params(which='both', right='off', top='off', direction='out', width=1)
        axes.yaxis.set_tick_params(which='both', right='off', top='off', direction='out', width=1)

        # axes.spines['top'].set_visible(False)

    def criar_pasta_simulacao(self, base_dir="simulacoes"):
        """
        Cria uma pasta para salvar os gráficos de uma simulação.

        Args:
            base_dir (str): Diretório base onde as pastas de simulação serão criadas.

        Returns:
            str: Caminho completo da pasta criada.
        """
        # Cria o diretório base, se não existir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Cria uma subpasta com base no timestamp atual
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.pasta_base = os.path.join(base_dir, f"simulacao_{timestamp}")
        os.makedirs(self.pasta_base)

        return self.pasta_base

    def plotar(self, x, y_list, labels, xlabel, ylabel, titulo=None, cores=None, marcadores=None, estilos=None,
               legenda_loc='lower center', legenda_ncol=6, grid=False, salvar=None, subplots=False, line=None,
               figsize=None):
        """
        Função para criar gráficos com múltiplas séries ou subplots.

        Args:
            x (array-like): Valores do eixo x.
            y_list (list of array-like): Lista de séries para o eixo y.
            labels (list of str): Lista de rótulos para cada série.
            xlabel (str): Rótulo do eixo x.
            ylabel (str or list): Rótulo do eixo y. Se for lista, cada subplot recebe seu próprio ylabel.
            titulo (str, optional): Título do gráfico.
            cores (list of str, optional): Lista de cores para cada série.
            marcadores (list of str, optional): Lista de marcadores para cada série.
            estilos (list of str, optional): Lista de estilos de linha para cada série.
            legenda_loc (str, optional): Localização da legenda.
            legenda_ncol (int, optional): Número de colunas na legenda.
            grid (bool, optional): Exibir grade no gráfico.
            salvar (str, optional): Nome do arquivo para salvar o gráfico (ex.: "grafico.png").
            subplots (bool or str, optional): Se True ou 'vertical', cria subplots verticais. Se 'horizontal', cria subplots lado a lado.
        """
        if subplots == 'horizontal':
            fig, axs = plt.subplots(1, len(y_list), figsize=figsize, sharex=True)
        elif subplots is True or subplots == 'vertical':
            fig, axs = plt.subplots(len(y_list), 1, figsize=figsize, sharex=True)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            axs = [ax]

        if cores is None:
            cores = ['black'] * len(y_list)
        if marcadores is None:
            marcadores = [''] * len(y_list)
        if estilos is None:
            estilos = ['-'] * len(y_list)

        if subplots:
            if len(y_list) == 1:
                axs = [axs]  # Garantir que axs seja uma lista mesmo com um único subplot
            for i, ax in enumerate(axs):
                # Se y_list[i] for uma lista de curvas, plote cada uma
                for j, y in enumerate(y_list[i]):
                    cor = cores[i][j] if isinstance(cores[i], (list, tuple, np.ndarray)) else cores[i]
                    marcador = marcadores[i][j] if isinstance(marcadores[i], (list, tuple, np.ndarray)) else marcadores[i]
                    estilo = estilos[i][j] if isinstance(estilos[i], (list, tuple, np.ndarray)) else estilos[i]
                    lbl = labels[i][j] if isinstance(labels[i], (list, tuple, np.ndarray)) else labels[i]
                    lwj = line[i][j] if (line is not None and isinstance(line[i], (list, tuple, np.ndarray))) else 2
                    ax.plot(x, y, color=cor, marker=marcador, linestyle=estilo, label=lbl, linewidth=lwj)
                
                # ylabel pode ser string ou lista
                ylab = ylabel[i] if isinstance(ylabel, (list, tuple)) else ylabel
                ax.set_ylabel(ylab, fontsize=self.fontsize)
                ax.tick_params(axis='both', labelsize=self.fontsize)
                
                # Legenda apenas no primeiro subplot
                if i == 0:
                    ax.legend(loc=legenda_loc, bbox_to_anchor=(0.5, 1), fontsize=self.fontsize, frameon=False, ncol=legenda_ncol)
                
                Plot.config_plot(ax)
                if grid:
                    ax.grid()
            axs[-1].set_xlabel(xlabel, fontsize=self.fontsize)
            if titulo:
                fig.suptitle(titulo, fontsize=self.fontsize)
            plt.tight_layout()
        else:
            for i, y in enumerate(y_list):
                lw = line[i] if isinstance(line, (list, tuple, np.ndarray)) else 2
                axs[0].plot(x, y, color=cores[i], marker=marcadores[i],
                            linestyle=estilos[i], label=labels[i], linewidth=lw)
            axs[0].set_xlabel(xlabel, fontsize=self.fontsize)
            axs[0].xaxis.set_major_locator(plt.MultipleLocator(100))
            axs[0].set_ylabel(ylabel, fontsize=self.fontsize)
            if titulo:
                axs[0].set_title(titulo, fontsize=self.fontsize)
            axs[0].tick_params(axis='both', labelsize=self.fontsize)
            axs[0].legend(loc=legenda_loc, bbox_to_anchor=(0.5, 1),
                          ncol=legenda_ncol, fontsize=self.fontsize, frameon=False)
            Plot.config_plot(axs[0])
            if grid:
                axs[0].grid()

        if salvar:
            # Adicionar a pasta base ao caminho, se definida
            if self.pasta_base:
                salvar = os.path.join(self.pasta_base, salvar)

            # Criar a pasta, se necessário
            pasta = os.path.dirname(salvar)
            if pasta and not os.path.exists(pasta):
                os.makedirs(pasta)

            # Salvar o gráfico
            plt.savefig(salvar, dpi=600, bbox_inches='tight')

