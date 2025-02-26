from manim import *
import numpy as np
from astropy.table import Table
from manim.utils.rate_functions import smooth

# Custom tail that only shows the most recent positions over a given time window.
class TracingTail(VMobject):
    def __init__(self, dot, time_traced=3, **kwargs):
        super().__init__(**kwargs)
        self.dot = dot
        self.time_traced = time_traced
        self.start_time = None
        self.points_with_time = []
        # Start with the dot's current position.
        self.set_points_as_corners([dot.get_center(), dot.get_center()])
        self.add_updater(self.update_tail)
    
    def update_tail(self, mob, dt):
        scene = mob.get_scene()
        if scene is None:
            return mob
        if self.start_time is None:
            self.start_time = scene.time
        current_time = scene.time
        # Append the current dot position with its timestamp.
        self.points_with_time.append((current_time, self.dot.get_center()))
        # Only keep positions from the last `time_traced` seconds.
        self.points_with_time = [
            (t, pt) for t, pt in self.points_with_time if current_time - t <= self.time_traced
        ]
        if self.points_with_time:
            pts = [pt for t, pt in self.points_with_time]
            mob.set_points_as_corners(pts)
        return mob

class OrbitWithTrails(ThreeDScene):
    def construct(self):
        # Load your orbital data. (Assumes columns "r0x", "r0y", "r0z", etc.)
        t_r_v = Table.read(
            '/Users/mncavieres/Documents/2024-2/HVS/Notebooks/manim/orbit_data.txt',
            format='ascii'
        )
        num_points = len(t_r_v["r0x"])
        
        # Compute dot0's median (fixed) position.
        dot0_median = np.array([
            np.median(t_r_v["r0x"]),
            np.median(t_r_v["r0y"]),
            np.median(t_r_v["r0z"])
        ])
        
        # Create three dots. dot0 is fixed (colored YELLOW for visibility on a dark background),
        # while dot1 and dot2 will move along their trajectories.
        dot0 = Dot3D(point=dot0_median, color=YELLOW, radius=0.2)
        dot1 = Dot3D(point=np.array([t_r_v["r1x"][0], t_r_v["r1y"][0], t_r_v["r1z"][0]]),
                     color=BLUE, radius=0.2)
        dot2 = Dot3D(point=np.array([t_r_v["r2x"][0], t_r_v["r2y"][0], t_r_v["r2z"][0]]),
                     color=GREEN, radius=0.2)
        self.add(dot0, dot1, dot2)
        
        # Create trailing tails for dot1 and dot2.
        tail1 = TracingTail(dot1, time_traced=3, stroke_color=BLUE, stroke_width=2)
        tail2 = TracingTail(dot2, time_traced=3, stroke_color=GREEN, stroke_width=2)
        self.add(tail1, tail2)
        
        # Create a ValueTracker to act as our time index.
        tracker = ValueTracker(0)
        
        # Updater for dot1 and dot2: they move along their orbits using the data.
        def update_dot(dot, x_key, y_key, z_key):
            def updater(mob, dt):
                i = int(tracker.get_value())
                i = min(i, num_points - 1)
                new_pos = np.array([
                    t_r_v[x_key][i],
                    t_r_v[y_key][i],
                    t_r_v[z_key][i]
                ])
                mob.move_to(new_pos)
            return updater
        
        dot1.add_updater(update_dot(dot1, "r1x", "r1y", "r1z"))
        dot2.add_updater(update_dot(dot2, "r2x", "r2y", "r2z"))
        
        # (Optional) Set up a moving camera. For example, let the camera follow the midpoint
        # of dot1 and dot2. You can modify this function to suit your desired transitions.
        def compute_camera_center(index):
            index = int(min(index, num_points - 1))
            p1 = np.array([t_r_v["r1x"][index],
                           t_r_v["r1y"][index],
                           t_r_v["r1z"][index]])
            p2 = np.array([t_r_v["r2x"][index],
                           t_r_v["r2y"][index],
                           t_r_v["r2z"][index]])
            return (p1 + p2) / 2
        
        dummy = Mobject()
        dummy.add_updater(lambda m, dt: setattr(self.camera, "frame_center",
                                                   compute_camera_center(tracker.get_value())))
        self.add(dummy)
        
        # Animate the orbit by increasing the tracker value.
        self.play(tracker.animate.set_value(num_points - 1), run_time=10, rate_func=linear)
        
        # Cleanup: remove updaters.
        dot1.clear_updaters()
        dot2.clear_updaters()
        tail1.clear_updaters()
        tail2.clear_updaters()
        dummy.clear_updaters()
