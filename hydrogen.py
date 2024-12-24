"""
Hydrogen CORRECT VERSION
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.optimize import bisect


N = 1000                      
r_max = 60                    
r = np.linspace(1e-5, r_max, N)  # (avoid r=0 for division)
dr = r[1] - r[0]              # Radial step size
l = 0                         # Angular momentum quantum number (s-states)
Z = 1                         # Atomic number (hydrogen atom)

# Effective potential for hydrogen atom
def effective_potential(r, l):
    return -Z / r + l * (l + 1) / (2 * r**2)


def solve_radial_schrodinger(E, l):
    V = effective_potential(r, l)
    k_squared = 2 * (E - V)
    u = np.zeros(N)
    
    # Boundary conditions for u(r)
    u[0] = 0
    u[1] = 1e-5  
    
    
    for i in range(1, N - 1):
        u[i + 1] = (2 * (1 - 5 * dr**2 / 12 * k_squared[i]) * u[i] -
                    (1 + dr**2 / 12 * k_squared[i - 1]) * u[i - 1]) / \
                   (1 + dr**2 / 12 * k_squared[i + 1])
        
        # Prevent overflow
        if np.isnan(u[i + 1]) or np.isinf(u[i + 1]):
            break
            
    return u

# Find eigenvalues using the shooting method with dynamic range adjustment
def find_energy_eigenvalue(l, energy_guess=(-1.2, -0.1)):
    def objective(E):
        u = solve_radial_schrodinger(E, l)
        return u[-2]  

    # Scan for sign change in the energy range
    E1, E2 = energy_guess
    while objective(E1) * objective(E2) > 0:  # No root yet
        E1 -= 0.1
        E2 += 0.1
        if E1 < -20:  # Prevent infinite loop
            raise ValueError("Could not bracket the root. Check the potential or initial guess.")

    # Use bisection method to find the eigenvalue
    energy_eigenvalue = bisect(objective, E1, E2, xtol=1e-6)
    return energy_eigenvalue


eigenvalues = []
wavefunctions = []

for n in range(1, 11):  # n ranges from 1 to 10
    energy_guess = (-0.5 / n**2 * 1.2, -0.5 / n**2 * 0.8)  # Adjust guesses near E_n = -1 / (2n^2)
    print(f"Finding eigenvalue for n = {n}...")
    
    E = find_energy_eigenvalue(l, energy_guess)
    eigenvalues.append(E)
    u = solve_radial_schrodinger(E, l)
    R = u / r  # Convert u(r) back to R(r)
    
    # Normalize wavefunction
    norm = simpson(R**2 * r**2, dx=dr)
    R_normalized = R / np.sqrt(norm)
    wavefunctions.append(R_normalized)

# Plot radial wavefunctions
plt.figure(figsize=(10, 8))
for n, R in enumerate(wavefunctions):
    plt.plot(r, R, label=f'n={n+1}, l={l}')
plt.xlabel('r [a_0]')
plt.ylabel('R(r)')
plt.title('Radial Wavefunctions for Hydrogen Atom (n=1 to 10, s-states (l=0))')
plt.legend()
plt.xlim(0,55)
plt.show()

# Print eigenvalues
print("\nComputed Eigenvalues (Hartree units):")
for n, E in enumerate(eigenvalues, start=1):
    print(f"n = {n}, E = {E:.6f}")
