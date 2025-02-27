from manim import *
import numpy as np
from astropy.table import Table

class OrbitAnimation(ThreeDScene):
    def construct(self):
        # Set the initial camera orientation for a 3D view.
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        # Load the orbital data from the text file using Table.read.
        t_r_v = Table.read(
            '/Users/mncavieres/Documents/2024-2/HVS/Notebooks/manim/orbit_data.txt',
            format='ascii'
        )
        num_points = len(t_r_v['r0x'])

        # Create Dot objects in 3D using the x, y, and z coordinates.
        dot0 = Dot(
            point=np.array([t_r_v['r0x'][0], t_r_v['r0y'][0], t_r_v['r0z'][0]]),
            color=WHITE, radius=0.3
        )
        dot1 = Dot(
            point=np.array([t_r_v['r1x'][0], t_r_v['r1y'][0], t_r_v['r1z'][0]]),
            color=BLUE, radius=0.2
        )
        dot2 = Dot(
            point=np.array([t_r_v['r2x'][0], t_r_v['r2y'][0], t_r_v['r2z'][0]]),
            color=GREEN, radius=0.2
        )
        self.add(dot0, dot1, dot2)

        # Create trails for dot1 and dot2 that trace their movement.
        trail1 = TracedPath(dot1.get_center, stroke_color=BLUE, stroke_width=5)
        trail2 = TracedPath(dot2.get_center, stroke_color=GREEN, stroke_width=5)
        self.add(trail1, trail2)

        # Create a ValueTracker that represents the time parameter.
        tracker = ValueTracker(0)

        # Updater function that moves a dot along its path in 3D by interpolating between data points.
        def update_dot(dot, key_x, key_y, key_z):
            t = tracker.get_value()
            index = int(np.floor(t))
            next_index = min(index + 1, num_points - 1)
            alpha = t - index
            # Interpolate the x, y, and z coordinates.
            x = (1 - alpha) * t_r_v[key_x][index] + alpha * t_r_v[key_x][next_index]
            y = (1 - alpha) * t_r_v[key_y][index] + alpha * t_r_v[key_y][next_index]
            z = (1 - alpha) * t_r_v[key_z][index] + alpha * t_r_v[key_z][next_index]
            dot.move_to(np.array([x, y, z]))

        # Attach updaters to dot1 and dot2.
        dot1.add_updater(lambda m, dt: update_dot(m, 'r1x', 'r1y', 'r1z'))
        dot2.add_updater(lambda m, dt: update_dot(m, 'r2x', 'r2y', 'r2z'))

        # Zoom out to show more of the 3D scene.
        self.play(self.camera.frame.animate.scale(2), run_time=2)

        # Animate the ValueTracker from 0 to num_points - 1 over 20 seconds.
        self.play(tracker.animate.set_value(num_points - 1), run_time=20, rate_func=linear)
        self.wait()
