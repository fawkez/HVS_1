from manim import *

class Hello(Scene):
    def construct(self):
        t = Text("Hello")
        self.play(Write(t))
        self.wait(2)

class First_scene(Scene):
    def construct(self):
        t = Text("This is the first scene")
        self.play(Write(t))
        self.wait(2)


class Second_scene(Scene):
    def construct(self):
        t = Text("This is the second scene")
        self.play(Write(t))
        self.wait(2)

class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))  # show the circle on screen