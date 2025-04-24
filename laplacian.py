import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve

# Parameters
N = 50  # Number of interior points per dimension
L = 1.0
h = L / (N + 1)  # Grid spacing

# Grid
x = np.linspace(h, L - h, N)
y = np.linspace(h, L - h, N)
X, Y = np.meshgrid(x, y)

# Right-hand side function f(x, y)
f = lambda x, y: 0.01/1e-12 * (x*0+1)
F = f(X, Y).reshape(N*N)

# Construct Laplacian operator with Dirichlet boundary conditions
main_diag = -4 * np.ones(N)
off_diag = np.ones(N - 1)
T = diags([main_diag, off_diag, off_diag], [0, -1, 1])
I = identity(N)
Laplacian = (kron(I, T) + kron(T, I)) / h**2

# Solve the linear system
U = spsolve(Laplacian, F)

# Reshape solution to 2D grid
U_grid = U.reshape((N, N))

# Plot the solution
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U_grid, cmap='viridis')
ax.set_title('Solution of ∇²u = f(x, y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x, y)')
plt.tight_layout()
plt.savefig("laplacian_solution.png", dpi=300)
plt.close() 
