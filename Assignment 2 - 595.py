import numpy as np
import matplotlib.pyplot as plt

def sin_cos_curve(period):
    π = np.pi
    time = np.arange(0, period * 2* π, 0.1)
    amplitud_sin = np.sin(time)
    amplitud_cos = np.cos(time)
    amplitud_tan = np.tan(time)

    fig, plot = plt.subplots()
    plt.plot(time, amplitud_sin, label = 'Sine')
    plt.plot(time, amplitud_cos, label = 'Cosine')
    plt.plot(time, amplitud_tan, label = 'Tan')
    plt.axhline(y = 0, color='black')
    
    xticks = np.arange(0, (period*2*π)+(π/2), π/2)
    xlabel = [int(np.rad2deg(x)) for x in xticks]
    
    plt.xticks(xticks)
    plot.set_xticklabels(xlabel)
    
    plot.set_xlim((0, period * 2 * np.pi)) 
    plot.set_ylim((-1, 1))
    
    plt.title(str(period)+'-Period: '+'Sin-Cos-tan', fontsize=20)
    plt.legend()
    plt.show()
    fig.savefig('sin_cos_tan.png')
sin_cos_curve(period = 2)
