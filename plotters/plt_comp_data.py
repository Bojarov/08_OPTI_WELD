import numpy as np
import matplotlib.pyplot as plt


# Plotting
def opt_plot_pre(p_opt_mat):
    freqs = [3, 6, 12, 24, 48, 96, 192, 384]

    fig = plt.figure(figsize=(9, 8))

    ax1 = fig.add_subplot(2, 4, 1)
    ax2 = fig.add_subplot(2, 4, 2)
    ax3 = fig.add_subplot(2, 4, 3)
    ax4 = fig.add_subplot(2, 4, 4)
    ax5 = fig.add_subplot(2, 4, 5)
    ax6 = fig.add_subplot(2, 4, 6)
    ax7 = fig.add_subplot(2, 4, 7)
    ax8 = fig.add_subplot(2, 4, 8)

    for f in freqs:
        f_ind = freqs.index(f)
        ax1.plot(p_opt_mat[:, f_ind, 0], label= 'f='+str(f))
        plt.grid()
        ax2.plot(p_opt_mat[:, f_ind, 1], label= 'f='+str(f))
        plt.grid()
        ax3.plot(p_opt_mat[:, f_ind, 2], label= 'f='+str(f))
        plt.grid()
        #ax4.plot(p_opt_mat[:, f_ind, 3], label= 'f='+str(f))
        #ax5.plot(p_opt_mat[:, f_ind, 4], label= 'f='+str(f))
        #ax6.plot(p_opt_mat[:, f_ind, 5], label= 'f='+str(f))
        #ax7.plot(p_opt_mat[:, f_ind, 6], label= 'f='+str(f))


    ax1.legend()

def opt_plot(p_opt_mat):
    freqs = [3, 6, 12, 24, 48, 96, 192, 384]
    freqs = [3, 6, 12, 24]

    fig = plt.figure(figsize=(9, 8))

    ax1 = fig.add_subplot(2, 4, 1)
    ax2 = fig.add_subplot(2, 4, 2)
    ax3 = fig.add_subplot(2, 4, 3)
    ax4 = fig.add_subplot(2, 4, 4)
    ax5 = fig.add_subplot(2, 4, 5)
    ax6 = fig.add_subplot(2, 4, 6)
    ax7 = fig.add_subplot(2, 4, 7)
    ax8 = fig.add_subplot(2, 4, 8)

    for f in freqs:
        f_ind = freqs.index(f)
        ax1.plot(p_opt_mat[:, f_ind, 0], label= 'f='+str(f))
        plt.grid()
        ax2.plot(p_opt_mat[:, f_ind, 1], label= 'f='+str(f))
        plt.grid()
        ax3.plot(p_opt_mat[:, f_ind, 2], label= 'f='+str(f))
        plt.grid()
        ax4.plot(p_opt_mat[:, f_ind, 3], label= 'f='+str(f))
        plt.grid()
        ax5.plot(p_opt_mat[:, f_ind, 4], label= 'f='+str(f))
        plt.grid()
        ax6.plot(p_opt_mat[:, f_ind, 5], label= 'f='+str(f))
        plt.grid()
        ax7.plot(p_opt_mat[:, f_ind, 6], label= 'f='+str(f))
        plt.grid()


    ax1.legend()