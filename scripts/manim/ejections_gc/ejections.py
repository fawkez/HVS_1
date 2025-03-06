from manim import *
import numpy as np
from astropy.table import Table

class OrbitAnimation_New(MovingCameraScene):
    def construct(self):
        # Load the orbit data saved as CSV.
        # Expecting columns: "x0", "z0", "x1", "z1", ... etc.
        t_r_v = Table.read(
            "/Users/mncavieres/Documents/2024-2/HVS/scripts/manim/ejections_gc/orbits.csv",
            format="csv"
        )
        num_points = len(t_r_v["x0"])  # assume all orbits have the same number of time steps

        # Create a static white dot at the origin.
        origin_dot = Dot(point=ORIGIN, color=WHITE)
        self.add(origin_dot)

        # Set the number of orbits (dots) to eject sequentially.
        n_orbits = 3
        delay = 3  # delay in tracker units between successive ejections
        
        # The maximum value of the global tracker is adjusted so that
        # the last dot gets a full orbit from its ejection time.
        max_tracker_value = (num_points - 1) + (n_orbits - 1) * delay

        # Create orbit dots and their trails.
        dots = []
        for i in range(n_orbits):
            # We assume the CSV has columns "x{i}" and "z{i}"
            dot = Dot(point=ORIGIN, color=WHITE)
            self.add(dot)
            dots.append(dot)
            # Add a trail to visualize the path
            self.add(TracedPath(dot.get_center, stroke_color=WHITE, stroke_width=1, dissipating_time= 2))

        # Create a global ValueTracker.
        tracker = ValueTracker(0)

        # Define an updater that ejects a dot only after its scheduled start time.
        def update_dot(dot, x_key, z_key, start_time):
            def updater(mob, dt):
                t_val = tracker.get_value()
                if t_val < start_time:
                    # Dot hasn't been ejected yet; keep it at the origin.
                    mob.move_to(ORIGIN)
                else:
                    # Compute effective time for this dot's orbit.
                    effective_t = t_val - start_time
                    idx = int(effective_t)
                    idx = min(idx, num_points - 1)
                    new_pos = np.array([t_r_v[x_key][idx], t_r_v[z_key][idx], 0])
                    mob.move_to(new_pos)
            return updater

        # Attach an updater to each dot with its own ejection delay.
        for i, dot in enumerate(dots):
            dot.add_updater(update_dot(dot, f"x{i}", f"z{i}", i * delay))

        # (Optional) Camera updater to keep all dots in view.
        def update_camera(frame, dt):
            positions = [origin_dot.get_center()] + [d.get_center() for d in dots]
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            center = np.array([(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0])
            frame.move_to(center)
            width = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.5
            frame.set_width(width)
        self.camera.frame.add_updater(update_camera)

        # Zoom out at the beginning.
        self.play(self.camera.frame.animate.scale(3), run_time=2)
        
        # Animate the global tracker from 0 to max_tracker_value.
        self.play(tracker.animate.set_value(max_tracker_value), run_time=10, rate_func=linear)

        # Clear updaters when the animation is done.
        for dot in dots:
            dot.clear_updaters()
        self.camera.frame.clear_updaters()
