import rebound
import numpy as np
import matplotlib.pyplot as plt

# Units: AU, Solar Masses, Days
# Gravitational constant in AU^3 / (M_sun * day^2)
G = 2.959122082855911e-4  

# Masses
M_bh = 4e6                # Black hole mass (in solar masses)
M1 = 2                    # Mass of star 1 (in solar masses)
M2 = 2                    # Mass of star 2 (in solar masses)
M_binary = M1 + M2        # Total mass of the binary

# Initial separation from the black hole
r_initial = 100           # in AU

# Separation between the two stars in the binary
a_binary = 0.1            # in AU

# Create a REBOUND simulation
sim = rebound.Simulation()
sim.units = ('AU', 'days', 'Msun')
sim.G = G

# Use a high-accuracy integrator suitable for close encounters
sim.integrator = "ias15"  # Adaptive, high-accuracy integrator

# Add the black hole at the origin
sim.add(m=M_bh)

# Compute the center of mass position and velocity of the binary
x_cm = r_initial
y_cm = 0.0
# Velocity for a parabolic trajectory towards the black hole
v_cm = np.sqrt(2 * G * M_bh / r_initial)
vx_cm = 0.0               # Moving along negative y-direction
vy_cm = -v_cm

# Compute the orbital speed of the binary stars around their center of mass
v_rel = np.sqrt(G * M_binary / a_binary)

# Positions of the stars relative to the binary center of mass
# Using reduced mass ratios
mu1 = M2 / M_binary
mu2 = M1 / M_binary
x1_rel, y1_rel = -mu1 * a_binary, 0.0
x2_rel, y2_rel = mu2 * a_binary, 0.0

# Velocities of the stars relative to the binary center of mass
# For circular orbit, velocities are perpendicular to the separation vector
v1_rel_x, v1_rel_y = 0.0, v_rel * mu1
v2_rel_x, v2_rel_y = 0.0, -v_rel * mu2

# Total positions and velocities of the stars
x1 = x_cm + x1_rel
y1 = y_cm + y1_rel
x2 = x_cm + x2_rel
y2 = y_cm + y2_rel

vx1 = vx_cm + v1_rel_x
vy1 = vy_cm + v1_rel_y
vx2 = vx_cm + v2_rel_x
vy2 = vy_cm + v2_rel_y

# Add the stars to the simulation
sim.add(m=M1, x=x1, y=y1, vx=vx1, vy=vy1)
sim.add(m=M2, x=x2, y=y2, vx=vx2, vy=vy2)

# Move to the center-of-momentum frame
sim.move_to_com()

# Set up arrays to store positions for plotting
times = []
x1_list = []
y1_list = []
x2_list = []
y2_list = []
x_bh_list = []
y_bh_list = []

# Simulation parameters
t_max = 5000  # in days
N_outputs = 1000
times = np.linspace(0, t_max, N_outputs)

# Integrate and store positions
for t in times:
    sim.integrate(t)
    particles = sim.particles
    x_bh_list.append(particles[0].x)
    y_bh_list.append(particles[0].y)
    x1_list.append(particles[1].x)
    y1_list.append(particles[1].y)
    x2_list.append(particles[2].x)
    y2_list.append(particles[2].y)

# Convert lists to arrays
x_bh_array = np.array(x_bh_list)
y_bh_array = np.array(y_bh_list)
x1_array = np.array(x1_list)
y1_array = np.array(y1_list)
x2_array = np.array(x2_list)
y2_array = np.array(y2_list)

# make a data frame and save to a csv file
import pandas as pd
#df = pd.DataFrame({'time': times, 'x_bh': x_bh_array, 'y_bh': y_bh_array, 'x1': x1_array, 'y1': y1_array, 'x2': x2_array, 'y2': y2_array})
#df.to_csv('simulation_data.csv', index=False)

# Plotting the trajectories
plt.figure(figsize=(10, 10))
plt.plot(x_bh_array, y_bh_array, 'ko', label='Black Hole')
plt.plot(x1_array, y1_array, 'b-', label='Star 1')
plt.plot(x2_array, y2_array, 'r-', label='Star 2')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title('Trajectories of Binary Stars Approaching a Black Hole')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
