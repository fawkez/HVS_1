from manim import *
import numpy as np
from astropy.table import Table

class OrbitAnimation(MovingCameraScene):
    def construct(self):
        # Load the orbital data from a text file using Table.read.
        t_r_v = Table.read(
            '/Users/mncavieres/Documents/2024-2/HVS/Notebooks/manim/orbit_data.txt',
            format='ascii'
        )
        num_points = len(t_r_v['r0x'])

        # Create Dot objects for the black hole and stars.
        # Here, dot0 is for the black hole and remains static.
        dot0 = Dot(point=np.array([t_r_v['r0x'][0], t_r_v['r0y'][0], 0]), color=WHITE)
        dot1 = Dot(point=np.array([t_r_v['r1x'][0], t_r_v['r1y'][0], 0]), color=BLUE)
        dot2 = Dot(point=np.array([t_r_v['r2x'][0], t_r_v['r2y'][0], 0]), color=GREEN)
        self.add(dot0, dot1, dot2)

        # Create trails for the moving dots (dot1 and dot2).
        trail1 = TracedPath(dot1.get_center, stroke_color=BLUE, stroke_width=2)
        trail2 = TracedPath(dot2.get_center, stroke_color=GREEN, stroke_width=2)
        self.add(trail1, trail2)

        # Create a ValueTracker that will serve as our "time" parameter.
        tracker = ValueTracker(0)

        # Updater function to move a dot along its path based on the tracker.
        def update_dot(dot, key_x, key_y):
            t = tracker.get_value()
            # Determine current index and the next index for interpolation.
            index = int(np.floor(t))
            next_index = min(index + 1, num_points - 1)
            alpha = t - index
            # Linear interpolation between consecutive points.
            x = (1 - alpha) * t_r_v[key_x][index] + alpha * t_r_v[key_x][next_index]
            y = (1 - alpha) * t_r_v[key_y][index] + alpha * t_r_v[key_y][next_index]
            dot.move_to(np.array([x, y, 0]))

        # Attach updaters to dot1 and dot2.
        dot1.add_updater(lambda m, dt: update_dot(m, 'r1x', 'r1y'))
        dot2.add_updater(lambda m, dt: update_dot(m, 'r2x', 'r2y'))

        self.play(self.camera.frame.animate.scale(4), run_time=2)

        # attach the camera to the midpoint of the two moving dots
        # Animate the ValueTracker from 0 to num_points - 1 over 10 seconds.
        self.play(tracker.animate.set_value(num_points - 1), run_time=10, rate_func=linear)
        self.wait()
