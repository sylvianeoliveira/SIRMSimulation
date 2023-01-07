from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import io
import base64

# Create your views here.
def home(request):
    taxas = {}
    N = request.GET.get("qtd_populacao", 500000)
    I = request.GET.get("qtd_infectados", 1)
    beta = request.GET.get("taxa_infeccao", 0.0000003)
    gama = request.GET.get("taxa_recuperacao", 0.1)
    m = request.GET.get("taxa_morte", 0.015)
    if (I > N):
        print("invalido")
    # grafica(beta, gama, m, N, I)
    return render(request, 'index.html', {
        "taxas": taxas
    })

def test(request):
    plt.plot(range(10))
    fig = plt.gcf()
    flike = io.BytesIO()
    fig.savefig(flike)
    b64 = base64.b64encode(flike.getvalue()).decode()
    return render(request, 'result/result.html', {'data': b64})

def RK4(tk, xk, h, F):
    """
    Implementa um passo de simulação do método clássico de Runge-Kutta
    de ordem 4, retornando o valor de x no passo seguinte.

    tk    -> instante de tempo atual
    xk    -> variável de interesse
    h     -> passo de simulação
    F     -> função F(tk, xk) tal que F(t, x) = dx/dt
    """

    m1 = F(tk      , xk)
    m2 = F(tk + h/2, xk + h/2*m1)
    m3 = F(tk + h/2, xk + h/2*m2)
    m4 = F(tk + h  , xk + h*m3)

    return xk + h/6*(m1 + 2*m2 + 2*m3 + m4)

def dSIRM(beta, gama, m):
    """
    Implementa a função vetorial F que satisfaz as equações diferenciais
    do modelo SIRM, tais que F(t, x) = dx/dt

    t     -> tempo em dias 
    x     -> vetor de dimensão 4 que representa as parcelas da população SIRM
    beta  -> taxa de contato/transmissão
    gama  -> taxa de recuperação
    m     -> taxa de mortalidade
    """

    def F_aux(t, x):
        f1 = -beta*x[0]*x[1]
        f2 = beta*x[0]*x[1] - gama*x[1] - m*x[1]
        f3 = gama*x[1]
        f4 = m*x[1]

        return np.array([f1, f2, f3, f4])
    
    return F_aux

def simulacao(tMax, x0, h, beta, gama, m):
    """
    Retorna os vetores t e x resultantes de uma simulação do modelo SIRM
    utilizando o método clássico de Runge-Kutta de ordem 4

    tMax    -> tempo a ser simulado
    x0      -> estado inicial da população
    h       -> passo de simulação
    beta  -> taxa de contato/transmissão
    gama  -> taxa de recuperação
    m     -> taxa de mortalidade
    """

    t = np.arange(0, tMax+h, h)
    x = np.zeros((4,len(t)))

    x[:,0] = x0
    F = dSIRM(beta, gama, m)
    
    for k in range(len(t)-1):
        x[:,k+1] = RK4(t[k], x[:,k], h, F)

    return t, x

def plot(axs, t, x):
    axs[0].plot(t, x[0,:])
    axs[1].plot(t, x[1,:])
    axs[2].plot(t, x[2,:])
    axs[3].plot(t, x[3,:])

def grafica(beta, gama, m, N, I):
    axs = []
    for i in range(4):
        fig, ax = plt.subplots(figsize=(10,7))
        axs.append(ax)

    axs[0].set_title("População Suscetível (S)")
    axs[1].set_title("População Infectada (I)")
    axs[2].set_title("População Recuperada (R)")
    axs[3].set_title("População Morta (M)")

    tMax = 1000
    x0 = np.array([N-I, I, 0, 0])
    h = 0.1

    t, x = simulacao(tMax, x0, h, beta, gama, m)
    plot(axs, t, x, "")

    for ax in axs:
        ax.legend()
        ax.set_xlabel("Dias")
        ax.set_ylabel("Nº Pessoas")
        ax.grid()

    plt.show()
