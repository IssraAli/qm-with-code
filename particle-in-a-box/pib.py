import numpy as np
import matplotlib.pyplot as plt

class StateArray:
    def __init__(self, samples: int, dt, ic):
        '''
        1-D state array for 1-D Schrodinger equation. Homogenous Dirichlet condition on the boundaries

        samples - number of partitions of array
        s - ratio of dt to dx**2
        ic - initial condition, callable
        '''

        self.samples = float
        self.domain, self.dx = np.linspace(0, 1, samples, retstep= True)
        self.stateArray = ic(self.domain) # initialize state array to initial condition
        self.stateArray[0] = 0
        self.stateArray[-1] = 0
        self.len = len(self.stateArray)
        self.dt = dt
        self.s = dt/(self.dx**2)

        ###### creating matrices for time-stepping
        
        self.alpha = 0.5 * (1j* self.s)/(1 + 1j * self.s)
        self.beta = (1 - self.s * 1j)/(1 + self.s * 1j)

        # A matrix
        m1 = np.identity(self.len - 2) * self.alpha # subtract two - endpoints
        m1 = np.pad(m1, ((1, 1), (0, 2)))
        m2 = np.roll(m1, 2, axis = 1)
        self.A = m1 + m2

        # B matrix
        self.B = np.identity(self.len) * self.beta + self.A
        self.B[0,0] = 1
        self.B[-1, -1] = 1

        self.time_step_matrix = np.matmul(np.linalg.inv(np.identity(self.len) - self.A), self.B)

    def time_step(self): 
        self.stateArray = np.matmul(self.stateArray, self.time_step_matrix)

    def animate(self, iters):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.canvas.draw()
        plt.pause(0.1)
        plt.show(block = False)
        
        state = self.stateArray
        domain = self.domain
        re, = ax.plot(domain[1:-2], state.real[1:-2], label = 'real', alpha = 0.2)
        im, = ax.plot(domain[1:-2], state.imag[1:-2], label = 'imag', alpha = 0.2)
        prob, = ax.plot(domain[1:-2], (state.real**2 + state.imag**2)[1:-2], label = 'probability')
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([-2, 5])
        
        for i in range (0, iters):
            self.time_step()
            state = self.stateArray
            re.set_ydata(state.real[1:-2])
            im.set_ydata(state.imag[1:-2])
            prob.set_ydata((state.real**2 + state.imag**2)[1:-2])
            
            for t in ax.texts:
                t.set_visible(False)
            p_integral = np.trapz((state.real**2 + state.imag**2)[1:-2], x = self.domain[1:-2])
            ax.text(0.2, 0.2, str(p_integral))
            fig.canvas.draw()
            plt.pause(0.05)           

def identity(arr):
    return arr

def sine(arr):
    return np.sqrt(2)*np.sin(2 * np.pi * arr)

def square(arr):
    return np.where((0.25 < arr) & (arr < 0.7), 1, 0)

def gauss(arr):
    return np.sqrt((10/np.sqrt(np.pi))) * np.exp((-(arr - 0.5)**2)/0.02)

def visualize_matrix(arr):
    '''utility to help make sure the matrices look as expected'''
    fig, ax = plt.subplots()
    ax.imshow(arr.real)
    plt.show()

if __name__ == "__main__":

    State = StateArray(201, ic = gauss, dt = 0.0005)
    print(State.stateArray, '\n')
    #visualize_matrix(State.time_step_matrix)
    State.animate(iters = 10000)
