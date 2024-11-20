from manim import *
import numpy as np

# Define the scaling factor
AU = 1.496e13  # 1 Astronomical Unit in cm
SCALE_FACTOR = 1 / AU  # Scaling positions to AUs

class ThreeBodyAnimation(Scene):
    def construct(self):
        # Load the data
        data = np.genfromtxt('simulation_data.csv', delimiter=',', names=True)

        # Extract positions and times
        times = data['time']
        x_0 = data['x_0'] * SCALE_FACTOR
        y_0 = data['y_0'] * SCALE_FACTOR
        x_1 = data['x_1'] * SCALE_FACTOR
        y_1 = data['y_1'] * SCALE_FACTOR
        x_2 = data['x_2'] * SCALE_FACTOR
        y_2 = data['y_2'] * SCALE_FACTOR

        # Create dots for each body
        body0 = Dot(point=ORIGIN, color=WHITE).scale(0.5)  # Black hole
        body1 = Dot(point=ORIGIN, color=BLUE).scale(0.3)   # Star 1
        body2 = Dot(point=ORIGIN, color=RED).scale(0.3)    # Star 2

        # Paths to trace the motion
        path1 = VMobject(color=BLUE)
        path2 = VMobject(color=RED)

        # Initialize paths with starting positions
        initial_pos1 = np.array([x_1[0], y_1[0], 0])
        initial_pos2 = np.array([x_2[0], y_2[0], 0])
        path1.set_points_as_corners(initial_pos1)
        path2.set_points_as_corners(initial_pos2)

        # Add initial elements to the scene
        self.add(body0, body1, body2, path1, path2)

        # Animation loop
        for i in range(1, len(times)):
            # Compute new positions
            new_pos0 = np.array([x_0[i], y_0[i], 0])
            new_pos1 = np.array([x_1[i], y_1[i], 0])
            new_pos2 = np.array([x_2[i], y_2[i], 0])

            # Update paths
            path1.add_points_as_corners([new_pos1])
            path2.add_points_as_corners([new_pos2])

            # Move bodies to new positions
            body0.move_to(new_pos0)
            body1.move_to(new_pos1)
            body2.move_to(new_pos2)

            # Wait for a short duration
            self.wait(0.001)

        # Keep the final frame for a few seconds
        self.wait(2)



class NBodyAnimation(Scene):
    def construct(self):
        #  Set up the axes
        axes = ThreeDAxes(
            x_range = (-50, 50, 5),
            y_range = (-50, 50, 5),
            z_range = (-0, 50, 5),
       #     width = 16,
       #     height = 16,
       #     depth=8
        )
        axes.set_width(100)
        axes.center()

        #self.frame.reorient(43, 76, 1, IN, 10)
        self.add(axes)

        # Load the data
        data = np.genfromtxt('simulation_data.csv', delimiter=',', names=True)

         # Extract positions and times
        times = data['time']
        x_0 = data['x_0'] * SCALE_FACTOR
        y_0 = data['y_0'] * SCALE_FACTOR
        x_1 = data['x_1'] * SCALE_FACTOR
        y_1 = data['y_1'] * SCALE_FACTOR
        x_2 = data['x_2'] * SCALE_FACTOR
        y_2 = data['y_2'] * SCALE_FACTOR

        points = np.array([x_0, y_0, [0]*len(x_0)]).T

        curve = VMobject().set_points_as_corners(points)   
        self.play(curve)


class ContinuousMotion(Scene):
    def construct(self):
        func = lambda pos: np.sin(pos[0] / 2) * UR + np.cos(pos[1] / 2) * LEFT
        stream_lines = StreamLines(func, stroke_width=2, max_anchors_per_line=30)
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)