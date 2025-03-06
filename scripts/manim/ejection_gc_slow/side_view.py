from manim import *
import numpy as np
from astropy.table import Table

class OrbitAnimationGlow(MovingCameraScene):
    def construct(self):
        # Load orbit data from CSV.
        # The CSV is expected to have columns labeled like "x0", "z0", "x1", "z1", etc.
        t_r_v = Table.read(
            "/Users/mncavieres/Documents/2024-2/HVS/Data/orbits/gala_ejections/orbits_animation.csv",
            format="csv"
        )
        num_points = 400#len(t_r_v["x0"])  # assuming each orbit has the same number of time steps

        # Select which orbit (i.e. which pair of columns) to animate for the ejected dot.
        selected_index = 0  # Change this index to select a different orbit

        # Create a static dot at the center (0,0).
        center_dot = Dot(point=ORIGIN, color=WHITE)
        #self.add(center_dot)
    

        # Create the ejected dot.
        ejected_dot = Dot(point=ORIGIN, radius=0.05, color=BLUE)
        self.add(ejected_dot)

        # Add a traced path (trail) to follow the ejected dot.
        self.add(
            TracedPath(
                ejected_dot.get_center, 
                stroke_color=BLUE, 
                stroke_width=2#, 
                #dissipating_time=1
            )
        )
        
        # Adjust the camera so that the entire trajectory fits in the frame.
        # Compute the bounding box for the selected orbit trajectory including the origin.
        x_vals = [0] + [float(x) for x in t_r_v[f"x{selected_index}"]]
        y_vals = [0] + [float(z) for z in t_r_v[f"z{selected_index}"]]
        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)
        frame_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, 0])
        # Calculate a width that fits the larger span of x or y coordinates, then add a margin.
        frame_width = max(max_x - min_x, max_y - min_y) * 1.5
        frame_width = max(frame_width, 9)   # enforce a minimum width
        frame_width = min(frame_width, 450) # enforce a maximum width
        #self.camera.frame.set_width(5)
        
        # move the camera so that the center of origin is in the lower right corner

        self.camera.frame.move_to(np.array([1,1,0]))

        # add axes
        axes = Axes(
            x_range=[0, 45, 5],
            y_range=[0, 30, 5],
            #axis_config={"color": WHITE},
            axis_config={"include_numbers": True},
            tips=False
        )

        labels = axes.get_axis_labels(
            Tex("X [kpc]").scale(0.7), Text("Z [kpc]").scale(0.45)
        )
        self.add(axes, labels)

        # Create a ValueTracker to drive the animation.
        tracker = ValueTracker(0)

        scale_factor_orbit = 3
        offset_x = 6
        offset_y = 3
        # Updater to move the ejected dot along its orbit.
        def update_ejected_dot(mob, dt):
            t_val = tracker.get_value()
            idx = int(t_val)
            idx = min(idx, num_points - 1)
            new_pos = np.array([
                float(t_r_v[f"x{selected_index}"][idx]/scale_factor_orbit) - offset_x,
                float(t_r_v[f"z{selected_index}"][idx]/scale_factor_orbit) - offset_y,
                0
            ])
            mob.move_to(new_pos)

        origin = np.array([
                float(t_r_v[f"x{selected_index}"][0]/scale_factor_orbit) - offset_x,
                float(t_r_v[f"z{selected_index}"][0]/scale_factor_orbit) - offset_y,
                0
            ])

        ejected_dot.add_updater(update_ejected_dot)

        # Optionally, zoom out initially.
        self.play(self.camera.frame.animate.scale(1.2), run_time=2)
        
        # Animate the ValueTracker from 0 to the last time step.
        self.play(tracker.animate.set_value(num_points - 1), run_time=5, rate_func=linear)

        # draw a vector from the origin to the ejected dot
        vector = Arrow(start=origin, end=ejected_dot.get_center(), color=TEAL)
        self.play(Create(vector), run_time = 2)

        # draw a vector from the ejected dot with a length of 1 and the velocity vector direction
        v_angle = 5 # degrees
        v_angle = np.radians(v_angle)
        length_velocity = 2
        # define end position
        end_arrow_velocity = ejected_dot.get_center() + length_velocity*np.cos(v_angle) * RIGHT +  length_velocity * np.sin(v_angle) * UP
        v_vector = Arrow(start=ejected_dot.get_center(), end=end_arrow_velocity, color=RED)
        self.play(Create(v_vector), run_time = 2)

        # change the v_vector to point paralel to the the other vector
        #self.play(v_vector.animate.rotate(v_angle, about_point=ejected_dot.get_center()), run_time = 2)

        self.wait(2)

        
        # Clean up updaters after the animation.
        ejected_dot.clear_updaters()
