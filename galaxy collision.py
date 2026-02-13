import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================================
# N-Body Simulation: Galaxy Collision
# RK4 integration with softened gravity
# ==========================================================

N_particles = 300
R_max = 10.0
H = 0.4
G = 1.0
M_gal = 200.0
M_central = 3000.0
eps = 0.5
sigma_frac = 0.03
dt = 0.01
frames = 600
offset = 25.0
v_rel = 5.2

# ----------------------------------------------------------
# Galaxy initialization
# ----------------------------------------------------------
def make_galaxy(N):
    m = 10**np.random.uniform(0,1,N)
    m *= M_gal/np.sum(m)

    u = np.random.rand(N)
    r_radius = R_max*np.sqrt(u)
    theta = 2*np.pi*np.random.rand(N)

    x = r_radius*np.cos(theta)
    y = r_radius*np.sin(theta)
    z = np.random.normal(0,H,N)

    r = np.stack([x,y,z],axis=1)

    v = np.zeros_like(r)
    for i in range(N):
        R = r_radius[i]
        if R > 1e-8:
            v_circ = np.sqrt(G*(M_central)/ (R+eps))
        else:
            v_circ = 0.0

        vx = -v_circ*np.sin(theta[i])
        vy =  v_circ*np.cos(theta[i])
        vz = 0.0

        vx += np.random.randn()*sigma_frac*v_circ
        vy += np.random.randn()*sigma_frac*v_circ
        vz += np.random.randn()*0.5*sigma_frac*v_circ

        v[i] = [vx,vy,vz]

    r -= np.average(r,axis=0,weights=m)
    v -= np.average(v,axis=0,weights=m)

    return r,v,m

# Create galaxies
r1,v1,m1 = make_galaxy(N_particles)
r2,v2,m2 = make_galaxy(N_particles)

r1[:,0] -= offset
r2[:,0] += offset
v1[:,0] += v_rel
v2[:,0] -= v_rel

r = np.vstack([r1,r2])
v = np.vstack([v1,v2])
m = np.concatenate([m1,m2])
colors = np.array(["cyan"]*N_particles + ["orange"]*N_particles)

# ----------------------------------------------------------
# Gravitational acceleration
# ----------------------------------------------------------
def acceleration(r):
    N = r.shape[0]
    a = np.zeros_like(r)

    for i in range(N):
        diff = r - r[i]
        dist2 = np.sum(diff**2,axis=1) + eps**2
        inv_dist3 = dist2**(-1.5)
        inv_dist3[i] = 0.0
        a[i] = G*np.sum((m[:,None]*diff)*inv_dist3[:,None],axis=0)

    return a

# ----------------------------------------------------------
# RK4 integrator
# ----------------------------------------------------------
def rk4_step(r,v,dt):
    def deriv(r_local,v_local):
        return v_local, acceleration(r_local)

    k1_r,k1_v = deriv(r,v)
    k2_r,k2_v = deriv(r+0.5*dt*k1_r, v+0.5*dt*k1_v)
    k3_r,k3_v = deriv(r+0.5*dt*k2_r, v+0.5*dt*k2_v)
    k4_r,k4_v = deriv(r+dt*k3_r, v+dt*k3_v)

    r_next = r + dt/6*(k1_r+2*k2_r+2*k3_r+k4_r)
    v_next = v + dt/6*(k1_v+2*k2_v+2*k3_v+k4_v)

    return r_next, v_next

# ----------------------------------------------------------
# Visualization
# ----------------------------------------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2*offset,2*offset)
ax.set_ylim(-2*offset,2*offset)
ax.set_zlim(-H*30,H*30)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.set_title("Galaxy Collision - N Body RK4", color='white')

scat = ax.scatter(r[:,0], r[:,1], r[:,2], s=6, c=colors)

def update(frame):
    global r,v
    r,v = rk4_step(r,v,dt)
    scat._offsets3d = (r[:,0],r[:,1],r[:,2])
    return scat,

ani = FuncAnimation(fig, update, frames=frames, interval=30, blit=False)
plt.show()
