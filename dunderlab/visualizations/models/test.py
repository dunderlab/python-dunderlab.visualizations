import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def get_cube(a, b, c, A=0, B=0, C=0):
    phi = np.arange(1, 10, 2) * np.pi / 4
    Phi, Theta = np.meshgrid(phi, phi)

    print(phi, Phi)

    x = np.cos(Phi) * np.sin(Theta)
    y = np.sin(Phi) * np.sin(Theta)
    z = np.cos(Theta) / np.sqrt(2)
    return [
        (x * a) - (x / 2) + A,
        (y * b) - (y / 2) + B,
        (z * c) - (z / 2) + C]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')


ax.plot_surface(*get_cube(1, 10, 10))
ax.plot_surface(*get_cube(1, 5, 5, A=1))


x, y, z = get_cube(1, 5, 5)

# ax.plot_surface(x, y, z, alpha=0.5)


ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
plt.axis('off')


plt.xlabel('X')
plt.ylabel('Y')

plt.show()
