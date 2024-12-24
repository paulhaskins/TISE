import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import simpson


# Parameters
N = 1000                    
gamma_squared = 200         
accuracy = 1e-6             # Desired accuracy for psi[N-1] = 0
delta_epsilon = 0.01        # Initial increment for epsilon
initial_epsilon = -0.99      # Starting trial energy below the expected eigenvalue

# Spatial grid
x = np.linspace(0, 1, N)    # Normalized x from 0 to 1
dx = x[1] - x[0]            

# Define potential array for the well
potential = -1 * np.ones(N)  # V(x) = -1 within the well

 
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
    
    #psi = wave_function(epsilon)
    return epsilon, psi_new





# first 10 energy eigenvalues and eigenstates
eigenvalues = []
eigenstates = []

epsilon = initial_epsilon
#WAS IN RANGE 10! XXXXXXX
for n in range(10):
    eigenvalue, eigenstate = find_eigenvalue(epsilon, delta_epsilon)
    eigenvalues.append(eigenvalue)
    eigenstates.append(eigenstate)
    epsilon = eigenvalue + 0.01 # Increment epsilon for next search


# Normalize the eigenfunctions using Simpson's rule
normalized_eigenstates = []
for psi in eigenstates:
    norm_factor = simpson(psi**2, dx=dx)
    normalized_psi = psi / np.sqrt(norm_factor)
    normalized_eigenstates.append(normalized_psi)

# Compare eigenvalues with analytic solution (for comparison only)
analytic_eigenvalues = [(np.pi * (n + 1))**2 for n in range(10)]

print(normalized_eigenstates)

#print("Computed eigenvalues:", eigenvalues)
#print("Analytic eigenvalues:", analytic_eigenvalues)

# Plot normalized eigenfunctions
plt.figure(figsize=(10, 6))
for n, psi in enumerate(normalized_eigenstates):
    plt.plot(x, psi, label=f"Eigenstate {n+1} (ε ≈ {eigenvalues[n]:.6f})")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title(" Wave Functions ψ(x) for ten Eigenstates (normalised)")
plt.legend()
plt.show()










# Compute uncertainty relation for each eigenstate
uncertainties = []


for psi in normalized_eigenstates:
    # Calculate <x> and <x^2>
    x_mean = simpson(x * psi**2, dx=dx)
    x_squared_mean = simpson(x**2 * psi**2, dx=dx)
    delta_x = np.sqrt(x_squared_mean - x_mean**2)
    
    # Calculate <p^2>
    psi_double_prime = np.zeros(N)
    for i in range(1, N - 1):
        psi_double_prime[i] = (psi[i-1] - 2*psi[i] + psi[i+1]) / dx**2
    
    p_squared_mean = -simpson(psi * psi_double_prime, dx=dx)
    delta_p = np.sqrt(p_squared_mean)
    
    uncertainties.append(delta_x * delta_p)

# Plot the uncertainty relation for the first 10 eigenstates
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11),   uncertainties, color='r', marker='x', label=" ΔxΔp(x)")
plt.axhline(y=0.5, color='b', linestyle='--', label="Min uncertainty (1/2)")
plt.xlabel("nth Eigenstate ")
plt.ylabel("ΔxΔp")
plt.title("Uncertainty of 10 Eigenstates (Square Well)")
plt.legend()
plt.show()


