from manim import *

def create_animation(script):
    class MyScene(Scene):
        def construct(self):
            text = Text(script)
            self.play(Write(text))
            self.wait(2)

    scene = MyScene()
    scene.render()  # Remove the disable_caching argument
