import numpy as np
import math


def fibonacci_sphere(n_samples):
    """Distributes n_samples on a unit fibonacci_sphere"""
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)

    for i in range(n_samples):
        y = 1 - (i / float(n_samples - 1)) * 2
        radius = math.sqrt(1 - y * y)

        theta = phi * i

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)


def fibonacci_azimuths_and_elevations(n_samples, listener_forward = np.array([0,1,0]), listener_left = np.array([-1,0,0])):
    a = fibonacci_sphere(n_samples)
    
    listener_up = np.cross(listener_forward, listener_left)
    listener_basis = np.stack((listener_forward, listener_left, listener_up), axis=-1)

    listener_coordinates = a @ listener_basis
    
    azimuths = np.round(np.degrees(np.arctan2(listener_coordinates[:, 1], listener_coordinates[:, 0])), 2)
    elevations = np.round(np.degrees(np.arctan(listener_coordinates[:, 2]/np.linalg.norm(listener_coordinates[:, 0:2],axis=-1)+1e-8)),2)
    
    return azimuths, elevations