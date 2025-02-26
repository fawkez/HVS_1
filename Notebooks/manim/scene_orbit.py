from manim import *
import numpy as np
from astropy.table import Table

class OrbitAnimation(MovingCameraScene):
    def construct(self):
        # Load the orbital data
        t_r_v = Table.read('/Users/mncavieres/Documents/2024-2/HVS/Notebooks/manim/orbit_data.txt', format='ascii')
        num_points = len(t_r_v['r0x'])

        # Create Dot objects for the black hole and stars
        dot0 = Dot(point=np.array([t_r_v['r0x'][0], t_r_v['r0y'][0], 0]), color=WHITE)
        dot1 = Dot(point=np.array([t_r_v['r1x'][0], t_r_v['r1y'][0], 0]), color=BLUE)
        dot2 = Dot(point=np.array([t_r_v['r2x'][0], t_r_v['r2y'][0], 0]), color=GREEN)

        self.add(dot0, dot1, dot2)

        # Create a ValueTracker for the time index
        tracker = ValueTracker(0)

        # Updater function to update the dots' positions based on the current index
        def update_dot(dot, x_key, y_key):
            def updater(mob, dt):
                i = int(tracker.get_value())
                i = min(i, num_points - 1)
                new_pos = np.array([t_r_v[x_key][i], t_r_v[y_key][i], 0])
                mob.move_to(new_pos)
            return updater

        # Attach the updaters to each dot
        dot0.add_updater(update_dot(dot0, 'r0x', 'r0y'))
        dot1.add_updater(update_dot(dot1, 'r1x', 'r1y'))
        dot2.add_updater(update_dot(dot2, 'r2x', 'r2y'))

        # ----------------------------
        # Dynamic camera adjustment:
        # ----------------------------
        def update_camera(frame, dt):
            # List of moving dots
            dots = [dot0]
            # Get current positions of the dots
            xs = [dot.get_center()[0] for dot in dots]
            ys = [dot.get_center()[1] for dot in dots]
            # Compute the center of the bounding box
            center = np.array([(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0])
            # Determine the width required to enclose all dots (with some margin)
            required_width = max(max() - min(xs), max(ys) - min(ys)) * 1.3
            frame.move_to(center)
            frame.set_width(required_width)
        
        # Add the updater to the camera frame so it tracks the moving dots
        self.camera.frame.add_updater(update_camera)
        
        # Animate the points by increasing the tracker value
        self.play(tracker.animate.set_value(num_points - 1), run_time=10, rate_func=linear)

        # Optionally, remove updaters once the animation is done
        dot0.clear_updaters()
        dot1.clear_updaters()
        dot2.clear_updaters()
        self.camera.frame.clear_updaters()


class MovingDots(Scene):
    def construct(self):
        d1,d2=Dot(color=BLUE),Dot(color=GREEN)
        dg=VGroup(d1,d2).arrange(RIGHT,buff=1)
        l1=Line(d1.get_center(),d2.get_center()).set_color(RED)
        x=ValueTracker(0)
        y=ValueTracker(0)
        d1.add_updater(lambda z: z.set_x(x.get_value()))
        d2.add_updater(lambda z: z.set_y(y.get_value()))
        l1.add_updater(lambda z: z.become(Line(d1.get_center(),d2.get_center())))
        self.add(d1,d2,l1)
        self.play(x.animate.set_value(5))
        self.play(y.animate.set_value(4))
        self.wait()
