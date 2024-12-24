"""
Correct Harmonic Potential
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Parameters
N = 1000                    
gamma_squared = 200         
accuracy = 1e-6             # Desired accuracy for psi[N-1] = 0
delta_epsilon = 0.01        # Initial increment for epsilon
initial_epsilon = -0.99     # Starting trial energy below the expected eigenvalue

# Spatial grid
x = np.linspace(0, 1, N)    # Normalized x from 0 to 1
dx = x[1] - x[0]            

# Define potential array for the given potential
potential = 8 * (x - 0.5)**2 - 1  # V(x) = 8(x - 0.5)^2 - 1

# Function to compute psi for a given epsilon
def wave_function(epsilon):
    psi = np.zeros(N)
    k_squared = gamma_squared * (epsilon - potential)
    psi[0] = 0
    psi[1] = 1e-4
    
    for i in range(1, N - 1):
        psi[i + 1] = (2 * (1 - (5/12) * dx**2 * k_squared[i]) * psi[i] - 
                      (1 + (1/12) * dx**2 * k_squared[i-1]) * psi[i-1]) / \
                     (1 + (1/12) * dx**2 * k_squared[i+1])
    
    return psi

# Function to find the eigenvalue using the shooting method
def find_eigenvalue(initial_epsilon, delta_epsilon):
    epsilon = initial_epsilon
    psi = wave_function(epsilon)
    psi_last = psi[-1]
    
    while abs(delta_epsilon) > accuracy:
        epsilon_new = epsilon + delta_epsilon
        psi_new = wave_function(epsilon_new)
        psi_last_new = psi_new[-1]
        
        if psi_last * psi_last_new < 0:
            delta_epsilon = -delta_epsilon / 2
        
        epsilon = epsilon_new
        psi_last = psi_last_new
    
    return epsilon, psi_new

# Compute the first 10 energy eigenvalues and eigenstates
eigenvalues = []
eigenstates = []

epsilon = initial_epsilon
for n in range(10):
    eigenvalue, eigenstate = find_eigenvalue(epsilon, delta_epsilon)
    eigenvalues.append(eigenvalue)
    eigenstates.append(eigenstate)
    epsilon = eigenvalue + 0.01  # Increment epsilon for next search

# Normalize the eigenfunctions using Simpson's rule
normalized_eigenstates = []
for psi in eigenstates:
    norm_factor = simpson(psi**2, dx=dx)
    normalized_psi = psi / np.sqrt(norm_factor)
    normalized_eigenstates.append(normalized_psi)

# Plot normalized eigenfunctions
plt.figure(figsize=(10, 6))
for n, psi in enumerate(normalized_eigenstates):
    plt.plot(x, psi, label=f"Eigenstate {n+1} (ε ≈ {eigenvalues[n]:.6f})")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title("Wave Functions in Harmonic Potential ψ(x) for Ten Eigenstates (Normalized)")
plt.legend()
plt.show()

# uncertainty relation for each eigenstate
uncertainties = []
for psi in normalized_eigenstates:
    #  <x> and <x^2>
    x_mean = simpson(x * psi**2, dx=dx)
    x_squared_mean = simpson(x**2 * psi**2, dx=dx)
    delta_x = np.sqrt(x_squared_mean - x_mean**2)
    
    # <p^2>
    psi_double_prime = np.zeros(N)
    for i in range(1, N - 1):
        psi_double_prime[i] = (psi[i-1] - 2*psi[i] + psi[i+1]) / dx**2
    
    p_squared_mean = -simpson(psi * psi_double_prime, dx=dx)
    delta_p = np.sqrt(p_squared_mean)
    
    uncertainties.append(delta_x * delta_p)

# Plot the uncertainty relation for the first 10 eigenstates
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), uncertainties, color='r', marker='x', label="ΔxΔp(x)")
plt.axhline(y=0.5, color='b', linestyle='--', label="Min uncertainty (1/2)")
plt.xlabel("nth Eigenstate")
plt.ylabel("ΔxΔp")
plt.title("Uncertainty of 10 Eigenstates")
plt.legend()
plt.show()

"""
t2
"""
# Extend eigenvalue computation to the first 20 eigenstates
eigenvalues = []
eigenstates = []

epsilon = initial_epsilon
for n in range(20):
    eigenvalue, eigenstate = find_eigenvalue(epsilon, delta_epsilon)
    eigenvalues.append(eigenvalue)
    eigenstates.append(eigenstate)
    epsilon = eigenvalue + 0.01  

# Compute the energy differences between adjacent eigenstates
energy_differences = [eigenvalues[n+1] - eigenvalues[n] for n in range(len(eigenvalues) - 1)]

# Plot the energy differences on a log-log scale
plt.figure(figsize=(8, 6))
plt.loglog(range(1, len(energy_differences) + 1), energy_differences, 'o-', color='r', label="Energy Differences")
plt.xlabel("log(n) ")
plt.ylabel("Energy Difference (log(ΔE))")
plt.title("Energy Differences Between Adjacent Eigenstates (Harmonic Potential in Log-Log Scale)")
plt.legend()

plt.show()



