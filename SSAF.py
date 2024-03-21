def plot_2d(matrix, title):
    if len(matrix.shape) == 1:  # Check if matrix is 1D
        matrix = matrix.reshape(-1, 1)  # Reshape to a 2D matrix
    height, width = matrix.shape
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.title(title)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def loadData(str):
    with open(str, "r") as file_solieu:
        data = np.loadtxt(file_solieu, dtype={'names': ('col1', 'col2'), 'formats': ('int', 'float')}, unpack=True)
    return data

def Step1(data, N, L):
    print(data[0][0])
    Fi = data[1][1:]
    # N = 100 # Number of signals to analyse
    # L = 50 # 2 < L < N - 1
    K = N - L + 1
    X = np.zeros((L, K)) # init X = (L, K)

    for i in range(L): # Rows
        for j in range(K): # Columns
            X[i, j] = Fi[i + j]

    U, Sigma, V = np.linalg.svd(X) # U = eigenvectors, V = processed vectors, Sigma = eigenvalues
    V = V.T

    d = 0
    for i in range(L):
        if Sigma[i] > 0:
            d = i + 1

    return (Fi, U, L, K, X, N, d)

def Step2(U, L, K, X, N, r):
    Pi = U
    PiT = np.transpose(Pi)

    # r = 3 # r is the linear space dimension, r < d
    Xmusum = np.zeros((L, K)) # Init Xmusum = sum of Xmu(i)
    for i in range(r):
        Xmusum += np.dot(np.outer(Pi[:, i], PiT[i, :]), X)

    Y = Xmusum # Substitute of matrix Xmusum = Y
    Lsao = min(L, K) # Init Lsao
    Ksao = max(L, K) # Init Ksao
    Ysao = np.zeros((L, K)) # Init Ysao
    if L < K: # When L < K then Ysao(i,j) = Y(i,j)
        Ysao = Y
    else: # When L >= K then Ysao(i,j) = Y(j,i)
        for i in range(Lsao):
            for j in range(K):
                Ysao[i, j] = Y[j, i]

    g = np.zeros(N) # Init array g with length N
    # Case 0 <= k < Lsao - 1 <=> 0 < h < Lsao
    for h in range(0, Lsao - 1):
        s = 0
        for m in range(0, h + 1): # Calculate sum Ysao
            s += Ysao[m, h - m]
        g[h] = (s / (h + 1))
    # Case Lsao - 1 <= k < Ksao <=> Lsao - 1 < h < Ksao + 1
    for h in range(Lsao - 1, Ksao): # sub h = k + 1
        s = 0
        for m in range(0, Lsao): # Calculate sum Ysao
            s += Ysao[m, h - m]
        g[h] = (s / (Lsao))
    # Case Ksao <= k < N => Ksao < h < N + 1
    for h in range(Ksao, N): # sub h = k + 1
        s = 0
        for m in range(h - Ksao + 1, N - K + 1): # Calculate sum Ysao
            s += Ysao[m, h - m]
        g[h] = (s / (N - h))

    return (Pi, g)

def Step3(r, Pi, g, L, N, M):
    nuy2 = 0 # Init nuy2
    for i in range(r):
        nuy2 += Pi[L - 1, i] ** 2 # Calculate nuy2 = π1^2 + . . . + πr^2

    # Calculate R using formula 2.1
    tempR = np.zeros(L - 1) # Init tempR
    for i in range(r): # Calculate sum Pi(L,i) * Pi(1:L-1,i)
        tempR += Pi[L - 1, i] * Pi[0:L - 1, i]

    R = 1 / (1 - nuy2) * tempR # Calculate R
    a = np.flipud(R) # Calculate a = matrix fliper of R

    M = 30 # Init extra steps we want to forecast
    gpredict = np.zeros(N + M) # Calculate g(i) with i=N+1:N+M
    for i in range(100):
        gpredict[i] = g[i]

    for i in range(N, N + M):
        gpredict[i] = 0
        for j in range(0, L - 1):
            gpredict[i] += (a[j] * gpredict[i - j - 1])

    return gpredict

def graphing(N, M, Fi, gpredict, save_path):
    matplotlib.use('agg')
    plt.figure()
    plt.plot(Fi, linewidth=1, linestyle='-', color='b', label='Fi') # Draw Fi
    plt.plot(gpredict, linewidth=1, linestyle='-', color='r', label='g') # Draw forecasting line
    plt.axvline(x=N, linewidth=0.5, linestyle='-', color='k', label='N line') #  y = N

    plt.xlabel('Time step') #  Time step
    plt.ylabel('Satellite signal') # Satellite signal
    plt.legend() # Output legend
    plt.xlim([0, N + M + 10]) # Limit x
    plt.xticks(range(0, N + M, 20)) # Limit y
    plt.savefig(save_path)  # Save the plot as an image
    plt.close() # Close the plot

    return save_path

def generate_inf_graph(fileName, r, M, N, L):
    data = loadData(fileName)
    limit = data[0][0]
    # r = 3
    # M = 30
    # N = 100
    # L = 20
    Fi, U, L, K, X, N, d = Step1(data, N, L)
    Pi, g = Step2(U, L, K, X, N, r)
    gpredict = Step3(r, Pi, g, L, N, M)
    save_path = 'static/Figure2.png'  # Define the path to save the image
    final_graph = graphing(N, M, Fi, gpredict, save_path)
    return final_graph

# def main():
#     data = loadData('uploads/dataset.txt')
#     Step1(data, 100, 50)

# if __name__ == "__main__":
#     main()
