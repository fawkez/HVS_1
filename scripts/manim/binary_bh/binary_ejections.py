from manim import *
import numpy as np
from astropy.table import Table

class OrbitBinaryBH(MovingCameraScene):
    def construct(self):
        # Load the orbital data from a text file using Table.read.
        t_r_v = Table.read(
            '/Users/mncavieres/Documents/2024-2/HVS/Notebooks/manim/orbit_data_binary_bh.txt',
            format='ascii'
        )
        num_points = len(t_r_v['r0x'])

        # Create Dot objects for the black hole and stars.
        # Here, dot0 is for the black hole and remains static.
        points_radius = 8
        dot0 = Dot(point=np.array([t_r_v['r0x'][0], t_r_v['r0y'][0], 0]), color=WHITE, radius=points_radius)
        dot1 = Dot(point=np.array([t_r_v['r1x'][0], t_r_v['r1y'][0], 0]), color=BLUE, radius=points_radius)
        dot2 = Dot(point=np.array([t_r_v['r2x'][0], t_r_v['r2y'][0], 0]), color=GREEN, radius = points_radius)
        self.add(dot0, dot1, dot2)

        trail_width = 120
        # Create trails for the moving dots (dot1 and dot2).
        trail1 = TracedPath(dot1.get_center, stroke_color=BLUE, stroke_width=trail_width)
        trail2 = TracedPath(dot2.get_center, stroke_color=GREEN, stroke_width=trail_width, dissipating_time=1.4)
        trail3 = TracedPath(dot0.get_center, stroke_color=WHITE, stroke_width=trail_width)
        self.add(trail1, trail2, trail3)

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
        dot0.add_updater(lambda m, dt: update_dot(m, 'r0x', 'r0y'))

        self.play(self.camera.frame.animate.scale(77), run_time=2)
        
           # --- Section 1: Animate until the first N points ---
        N = 380  # Change this value as needed (make sure N <= num_points)
        self.play(tracker.animate.set_value(N - 1), run_time=14, rate_func=linear)
        #self.wait(2)
        
        # --- Compute bounding box and zoom out to show all points ---
        # Combine all x and y data from the three orbits.
        all_x = np.concatenate([t_r_v['r0x'], t_r_v['r1x'], t_r_v['r2x']])
        all_y = np.concatenate([t_r_v['r0y'], t_r_v['r1y'], t_r_v['r2y']])
        min_x, max_x = np.min(all_x), np.max(all_x)
        min_y, max_y = np.min(all_y), np.max(all_y)
        # Compute the center and desired width (with extra margin).
        center =np.array([400, 0, 0]) #np.array([(min_x + max_x) / 2, (min_y + max_y)/ 2, 0])
        desired_width = (max_x - min_x) * 1  # Increase by 20% for margin
        print(desired_width, center)
        
        # Animate the camera frame to move to the center and adjust its width.
        self.play(
            self.camera.frame.animate.set(width=desired_width).move_to(center),
            run_time=2
        )
        #self.wait(2)
        
        # --- Section 2: Continue animation until the end ---
        self.play(tracker.animate.set_value(num_points - 300), run_time=20, rate_func=linear)
        self.wait()

    