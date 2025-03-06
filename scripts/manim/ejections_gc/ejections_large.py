from manim import *
import numpy as np
from astropy.table import Table

class OrbitLarger(MovingCameraScene):
    def construct(self):
        # Load the orbit data saved as CSV.
        # Expecting columns: "x0", "z0", "x1", "z1", ... "x1999", "z1999"
        t_r_v = Table.read(
            "/Users/mncavieres/Documents/2024-2/HVS/scripts/manim/ejections_gc/orbits.csv",
            format="csv"
        )
        num_points = len(t_r_v["x0"])  # assume all orbits have the same number of time steps
        
        # Create a static white dot at the origin.
        origin_dot = Dot(point=ORIGIN, color=WHITE)
        self.add(origin_dot)
        
        # Set the number of orbits to 2000.
        n_orbits = 10
        
        # Create orbit dots and their trails.
        dots = []
        for i in range(n_orbits):
            x_key = f"x{i}"
            z_key = f"z{i}"
            # Initial position from the CSV (mapping CSV x and z to scene x and y).
            initial_pos = np.array([t_r_v[x_key][0], t_r_v[z_key][0], 0])
            dot = Dot(point=initial_pos, color=WHITE)
            self.add(dot)
            dots.append(dot)
            # Add a trail for the dot.
            self.add(TracedPath(dot.get_center, stroke_color=WHITE, stroke_width=1))
        
        # Create a ValueTracker for the time index.
        tracker = ValueTracker(0)
        
        # Updater function: each dot reads its corresponding CSV columns based on the tracker's value.
        def update_dot(dot, x_key, z_key):
            def updater(mob, dt):
                i = int(tracker.get_value())
                # Ensure we don't go past the data length.
                i = min(i, num_points - 1)
                new_pos = np.array([t_r_v[x_key][i], t_r_v[z_key][i], 0])
                mob.move_to(new_pos)
            return updater
        
        # Attach updaters to each orbit dot.
        for i, dot in enumerate(dots):
            dot.add_updater(update_dot(dot, f"x{i}", f"z{i}"))
        
        # Optionally, add an updater to the camera frame to keep all dots in view.
        def update_camera(frame, dt):
            # Collect positions from the origin and all orbit dots.
            positions = [origin_dot.get_center()] + [d.get_center() for d in dots]
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            center = np.array([(min(xs)+max(xs))/2, (min(ys)+max(ys))/2, 0])
            frame.move_to(center)
            width = max(max(xs)-min(xs), max(ys)-min(ys)) * 1.5
            frame.set_width(width)

        #self.camera.frame.add_updater(update_camera)
        # zoom out
        #self.play(self.camera.frame.animate.scale(20), run_time=2)
        
        # Animate the tracker over the full range of time steps.
        self.play(tracker.animate.set_value(num_points - 1), run_time=10, rate_func=linear)
        
        # Clean up updaters.
        for dot in dots:
            dot.clear_updaters()
        self.camera.frame.clear_updaters()
