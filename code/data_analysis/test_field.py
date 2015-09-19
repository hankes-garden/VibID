import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    w = np.linspace(1.0, 100.0, 1000)
    
    wn = 35.0
    
    for xi in np.linspace(0.1, 0.9, 10):
        A = 1.0 / np.sqrt( (1-(w/wn)**2)**2 + (2*xi*w/wn)**2 )
        plt.plot(w, A, label=xi)
    plt.legend()
    plt.show()
    