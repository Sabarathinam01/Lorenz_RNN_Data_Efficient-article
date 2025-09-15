import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lorenz system parameters
sigma, rho, beta = 10.0, 28.0, 8.0/3.0

def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Integration settings
t0, t1 = 0, 40
t_eval = np.linspace(t0, t1, 10000)
init_state = [1.0, 1.0, 1.0]
sol = solve_ivp(lorenz, (t0, t1), init_state, t_eval=t_eval)

x, y, z = sol.y
t = sol.t

# Equilibria
eq1 = np.array([0.0, 0.0, 0.0])
x_eq = np.sqrt(beta*(rho-1)) if rho > 1 else 0
eq2 = np.array([ x_eq,  x_eq, rho-1])
eq3 = np.array([-x_eq, -x_eq, rho-1])
equilibria = [eq1, eq2, eq3]

# Create colormap along trajectory (same for all subplots)
colors = plt.cm.plasma(np.linspace(0, 1, len(t)))

# Plotting
fig = plt.figure(figsize=(16, 12))

# --- Time series ---
ax1 = fig.add_subplot(3,3,1)
ax1.scatter(t, x, c=colors, s=0.5)
ax1.set_title("x(t)")
ax1.set_xlabel("t"); ax1.set_ylabel("x")

ax2 = fig.add_subplot(3,3,2)
ax2.scatter(t, y, c=colors, s=0.5)
ax2.set_title("y(t)")
ax2.set_xlabel("t"); ax2.set_ylabel("y")

ax3 = fig.add_subplot(3,3,3)
ax3.scatter(t, z, c=colors, s=0.5)
ax3.set_title("z(t)")
ax3.set_xlabel("t"); ax3.set_ylabel("z")

# --- Phase portraits ---
ax4 = fig.add_subplot(3,3,4)
ax4.scatter(x, y, c=colors, s=0.5)
for e in equilibria:
    ax4.scatter(e[0], e[1], color="black", marker="o", s=40)
ax4.set_title("XY phase portrait")
ax4.set_xlabel("x"); ax4.set_ylabel("y")

ax5 = fig.add_subplot(3,3,5)
ax5.scatter(x, z, c=colors, s=0.5)
for e in equilibria:
    ax5.scatter(e[0], e[2], color="black", marker="o", s=40)
ax5.set_title("XZ phase portrait")
ax5.set_xlabel("x"); ax5.set_ylabel("z")

ax6 = fig.add_subplot(3,3,6)
ax6.scatter(y, z, c=colors, s=0.5)
for e in equilibria:
    ax6.scatter(e[1], e[2], color="black", marker="o", s=40)
ax6.set_title("YZ phase portrait")
ax6.set_xlabel("y"); ax6.set_ylabel("z")

# --- 3D Lorenz attractor ---
ax7 = fig.add_subplot(3,3,(7,9), projection="3d")
ax7.scatter(x, y, z, c=colors, s=0.5)
for e in equilibria:
    ax7.scatter(e[0], e[1], e[2], color="black", marker="o", s=50)
ax7.set_title("3D Lorenz Attractor")
ax7.set_xlabel("x"); ax7.set_ylabel("y"); ax7.set_zlabel("z")
ax7.view_init(elev=25, azim=135)

plt.tight_layout()
plt.show()

# Print equilibria
print("Equilibrium points:")
for i, e in enumerate(equilibria, start=1):
    print(f"E{i}: ({e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f})")
