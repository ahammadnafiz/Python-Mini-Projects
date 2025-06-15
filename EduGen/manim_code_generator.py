import os
import re
import json
import logging
from dotenv import load_dotenv
import textwrap  # Added to normalize indentation
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logging.getLogger('comtypes').setLevel(logging.WARNING)

load_dotenv('.env')

class ManIMCodeGenerator:
    def __init__(self, google_api_key):
        self.google_api_key = google_api_key
        self.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
        self.google_chat = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.google_api_key,
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        
        # Enhanced Manim code generation prompt
        self.manim_prompt = self._create_manim_generation_prompt()
        
        # Manim conversation chain
        self.manim_conversation = ConversationChain(
            llm=self.google_chat,
            prompt=self.manim_prompt,
            verbose=True,
            memory=self.memory,
            input_key="human_input",
        )

    def generate_3b1b_manim_code(self, video_plan):
        """
        Generate comprehensive, dynamic Manim code following 3Blue1Brown style.
        
        This creates step-by-step animations with:
        - Rich visual elements and smooth transitions
        - Dynamic positioning and movement
        - Scene progression without overlapping text
        - Educational flow based on the video plan structure
        
        Args:
            video_plan (dict): Complete video plan from script generator
            
        Returns:
            str: Complete Manim Python code ready for execution
        """
        if not video_plan:
            raise ValueError("No video plan provided")
            
        educational_breakdown = video_plan.get("educational_breakdown", {})
        manim_structure = video_plan.get("manim_structure", {})
        
        if not educational_breakdown:
            raise ValueError("No educational content available")
        
        try:
            print("🎨 Generating Advanced Manim Code...")
            print("📚 Topic: {}".format(educational_breakdown.get('title', 'Unknown')))
            print("🎯 Educational Steps: {}".format(len(educational_breakdown.get('educational_steps', []))))
            print("=" * 60)
            
            # Display video plan details in terminal
            self._display_video_plan(video_plan)
            
            # Build comprehensive prompt for Manim code generation
            manim_prompt = self._build_advanced_manim_prompt(video_plan)
            
            print("🔄 Processing with AI...")
            response = self.manim_conversation.predict(human_input=manim_prompt)
            
            # Extract and validate Manim code
            manim_code = self._extract_manim_code(response)
            
            # Validate and fix the code to remove image references
            if manim_code:
                manim_code = self._validate_and_fix_manim_code(manim_code)
            
            if manim_code:
                print("✅ Advanced Manim Code Generated Successfully!")
                print("📝 Code Length: {} characters".format(len(manim_code)))
                print("🎬 Ready for animation rendering!")
                
                # Display generated manim code in terminal
                self._display_manim_code(manim_code)
                
                return manim_code
            else:
                raise Exception("Code extraction failed")
                
        except Exception as e:
            print("❌ Error in Manim code generation: {}".format(e))
            raise

    def _build_advanced_manim_prompt(self, video_plan):
        """
        Build a comprehensive prompt for advanced Manim code generation.
        
        Args:
            video_plan (dict): Complete video plan with educational breakdown
            
        Returns:
            str: Detailed prompt for Manim code generation
        """
        educational_breakdown = video_plan.get("educational_breakdown", {})
        manim_structure = video_plan.get("manim_structure", {})
        
        title = educational_breakdown.get("title", "Educational Animation")
        steps = educational_breakdown.get("educational_steps", [])
        duration = educational_breakdown.get("metadata", {}).get("estimated_total_duration", 180)
        
        prompt_parts = ["""
ADVANCED MANIM CODE GENERATION REQUEST

VIDEO PLAN TO IMPLEMENT:
Title: {title}
Duration: {duration} seconds
Educational Steps: {steps_count}""".format(title=title, duration=duration, steps_count=len(steps)) + """

REQUIREMENTS FOR MANIM CODE:

🎯 EDUCATIONAL FLOW:
- Convert each educational step into a distinct scene method
- Maintain pedagogical progression from the video plan
- Use dynamic positioning and smooth transitions
- Create engaging visual storytelling

🎨 ANIMATION STYLE (3Blue1Brown Inspired):
- Rich visual elements with proper spacing
- Dynamic camera movements when appropriate
- Smooth object transformations and reveals
- Color-coded elements for better understanding
- Mathematical notation rendered clearly
- No overlapping text or crowded scenes

🏗️ CODE STRUCTURE REQUIREMENTS:
- Main scene class inheriting from Scene
- Separate methods for each educational step
- construct() method orchestrating the flow
- Proper imports and dependencies
- Clean, well-documented code
- Modular design for easy modification

🎬 ANIMATION TECHNIQUES:
- Use Write(), FadeIn(), Transform(), Create() appropriately
- Implement proper timing with self.wait()
- Position elements using UP, DOWN, LEFT, RIGHT vectors
- Scale and rotate objects for visual interest
- Use color schemes that enhance understanding
- Clear scene transitions between steps

📊 VISUAL ELEMENTS TO INCLUDE:
- Title animations with engaging reveals
- Step-by-step concept introductions
- Mathematical equations and formulas
- Diagrams and geometric shapes (using built-in Manim objects)
- Text labels and annotations
- Real-world example descriptions (text-based, NO ImageMobject)
- Summary and key takeaway displays

⚠️ CRITICAL CONSTRAINTS:
- DO NOT use ImageMobject or any image file references
- Use only text, shapes, and built-in Manim objects
- Create visual diagrams using Circle, Rectangle, Line, etc.
- Represent real-world examples with text descriptions and geometric visualizations
- Focus on mathematical notation, graphs, and animated text elements

⚡ DYNAMIC FEATURES & SCENE MANAGEMENT:
- Objects that move and transform with smooth transitions
- Reveal animations for key concepts (Write, FadeIn, Transform)
- Highlighting and emphasis effects (Indicate, Flash, Wiggle)
- Clear scene transitions - remove old content before adding new
- Use self.clear() or FadeOut() to clean scenes between steps
- Dynamic positioning - move objects to different locations
- Interactive-style demonstrations with step-by-step reveals
- Progressive complexity building with animated transformations
- Avoid static layouts - everything should move and change
- No overlapping text - use proper spacing and timing
- Create visual flow with object movements and morphing

🎯 MANDATORY POSITIONING RULES:
- NEVER place text in the same position (0,0) or CENTER
- Use UP, DOWN, LEFT, RIGHT with multipliers (2*UP, 3*LEFT, etc.)
- Position titles at 3*UP, subtitles at 2*UP, content at CENTER to DOWN
- Move previous content OFF-SCREEN before adding new content
- Use .shift(LEFT*4) or .shift(RIGHT*4) to move objects sideways
- Scale objects (.scale(0.8)) to fit more content without overlap
- Always animate movements: self.play(obj.animate.shift(UP*2))

🎬 REQUIRED ANIMATION PATTERNS:
- Start each section by clearing: self.play(FadeOut(*self.mobjects))
- Introduce titles with Write() animation
- Move titles up: self.play(title.animate.shift(UP*2))
- Add content below with different Y positions
- Use Transform() to change content, not create new overlapping text
- End sections with content moving off-screen or fading out

🎨 VISUAL VARIETY REQUIREMENTS:
- Use different font sizes: font_size=48 for titles, 36 for subtitles, 24 for content
- Use colors: BLUE for titles, WHITE for content, YELLOW for emphasis
- Create diagrams with Circle(), Rectangle(), Line() objects
- Position diagrams LEFT and text RIGHT, or vice versa
- Use arrows (Arrow()) to connect related concepts
- Create mathematical plots with axes when relevant

📚 PROFESSIONAL MANIM EXAMPLES GALLERY:
Study these high-quality examples for code patterns and techniques:

Example 1: BraceAnnotation - Annotating geometric elements
```python
from manim import *

class BraceAnnotation(Scene):
    def construct(self):
        dot = Dot([-2, -1, 0])
        dot2 = Dot([2, 1, 0])
        line = Line(dot.get_center(), dot2.get_center()).set_color(ORANGE)
        b1 = Brace(line)
        b1text = b1.get_text("Horizontal distance")
        b2 = Brace(line, direction=line.copy().rotate(PI / 2).get_unit_vector())
        b2text = b2.get_tex("x-x_1")
        self.add(line, dot, dot2, b1, b2, b1text, b2text)
```

Example 2: VectorArrow - Coordinate system visualization
```python
from manim import *

class VectorArrow(Scene):
    def construct(self):
        dot = Dot(ORIGIN)
        arrow = Arrow(ORIGIN, [2, 2, 0], buff=0)
        numberplane = NumberPlane()
        origin_text = Text('(0, 0)').next_to(dot, DOWN)
        tip_text = Text('(2, 2)').next_to(arrow.get_end(), RIGHT)
        self.add(numberplane, dot, arrow, origin_text, tip_text)
```

Example 3: BooleanOperations - Interactive shape operations
```python
from manim import *

class BooleanOperations(Scene):
    def construct(self):
        ellipse1 = Ellipse(
            width=4.0, height=5.0, fill_opacity=0.5, color=BLUE, stroke_width=10
        ).move_to(LEFT)
        ellipse2 = ellipse1.copy().set_color(color=RED).move_to(RIGHT)
        bool_ops_text = MarkupText("<u>Boolean Operation</u>").next_to(ellipse1, UP * 3)
        ellipse_group = Group(bool_ops_text, ellipse1, ellipse2).move_to(LEFT * 3)
        self.play(FadeIn(ellipse_group))

        i = Intersection(ellipse1, ellipse2, color=GREEN, fill_opacity=0.5)
        self.play(i.animate.scale(0.25).move_to(RIGHT * 5 + UP * 2.5))
        intersection_text = Text("Intersection", font_size=23).next_to(i, UP)
        self.play(FadeIn(intersection_text))

        u = Union(ellipse1, ellipse2, color=ORANGE, fill_opacity=0.5)
        union_text = Text("Union", font_size=23)
        self.play(u.animate.scale(0.3).next_to(i, DOWN, buff=union_text.height * 3))
        union_text.next_to(u, UP)
        self.play(FadeIn(union_text))
```

Example 4: PointMovingOnShapes - Path animations and transformations
```python
from manim import *

class PointMovingOnShapes(Scene):
    def construct(self):
        circle = Circle(radius=1, color=BLUE)
        dot = Dot()
        dot2 = dot.copy().shift(RIGHT)
        self.add(dot)

        line = Line([3, 0, 0], [5, 0, 0])
        self.add(line)

        self.play(GrowFromCenter(circle))
        self.play(Transform(dot, dot2))
        self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
        self.play(Rotating(dot, about_point=[2, 0, 0]), run_time=1.5)
        self.wait()
```

Example 5: MovingAround - Object transformations with animate
```python
from manim import *

class MovingAround(Scene):
    def construct(self):
        square = Square(color=BLUE, fill_opacity=1)

        self.play(square.animate.shift(LEFT))
        self.play(square.animate.set_fill(ORANGE))
        self.play(square.animate.scale(0.3))
        self.play(square.animate.rotate(0.4))
```

Example 6: MovingAngle - Dynamic angle measurement with updaters
```python
from manim import *

class MovingAngle(Scene):
    def construct(self):
        rotation_center = LEFT

        theta_tracker = ValueTracker(110)
        line1 = Line(LEFT, RIGHT)
        line_moving = Line(LEFT, RIGHT)
        line_ref = line_moving.copy()
        line_moving.rotate(
            theta_tracker.get_value() * DEGREES, about_point=rotation_center
        )
        a = Angle(line1, line_moving, radius=0.5, other_angle=False)
        tex = MathTex(r"\theta").move_to(
            Angle(
                line1, line_moving, radius=0.5 + 3 * SMALL_BUFF, other_angle=False
            ).point_from_proportion(0.5)
        )

        self.add(line1, line_moving, a, tex)
        self.wait()

        line_moving.add_updater(
            lambda x: x.become(line_ref.copy()).rotate(
                theta_tracker.get_value() * DEGREES, about_point=rotation_center
            )
        )

        a.add_updater(
            lambda x: x.become(Angle(line1, line_moving, radius=0.5, other_angle=False))
        )
        tex.add_updater(
            lambda x: x.move_to(
                Angle(
                    line1, line_moving, radius=0.5 + 3 * SMALL_BUFF, other_angle=False
                ).point_from_proportion(0.5)
            )
        )

        self.play(theta_tracker.animate.set_value(40))
        self.play(theta_tracker.animate.increment_value(140))
        self.play(tex.animate.set_color(RED), run_time=0.5)
        self.play(theta_tracker.animate.set_value(350))
```

Example 7: MovingDots - Connected objects with updaters
```python
from manim import *

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
```

Example 8: MovingFrameBox - Highlighting mathematical expressions
```python
from manim import *

class MovingFrameBox(Scene):
    def construct(self):
        self.play(Write(text))
        framebox1 = SurroundingRectangle(text[1], buff = .1)
        framebox2 = SurroundingRectangle(text[3], buff = .1)
        self.play(Create(framebox1))
        self.wait()
        self.play(ReplacementTransform(framebox1,framebox2))
        self.wait()
```

Example 9: SinAndCosFunctionPlot - Mathematical function plotting
```python
from manim import *

class SinAndCosFunctionPlot(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-10, 10.3, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=10,
            axis_config={"color": GREEN},
            x_axis_config={
                "numbers_to_include": np.arange(-10, 10.01, 2),
                "numbers_with_elongated_ticks": np.arange(-10, 10.01, 2),
            },
            tips=False,
        )
        axes_labels = axes.get_axis_labels()
        sin_graph = axes.plot(lambda x: np.sin(x), color=BLUE)
        cos_graph = axes.plot(lambda x: np.cos(x), color=RED)

        sin_label = axes.get_graph_label(
            sin_graph, "\\sin(x)", x_val=-10, direction=UP / 2
        )
        cos_label = axes.get_graph_label(cos_graph, label="\\cos(x)")

        vert_line = axes.get_vertical_line(
            axes.i2gp(TAU, cos_graph), color=YELLOW, line_func=Line
        )
        line_label = axes.get_graph_label(
            cos_graph, r"x=2\pi", x_val=TAU, direction=UR, color=WHITE
        )

        plot = VGroup(axes, sin_graph, cos_graph, vert_line)
        labels = VGroup(axes_labels, sin_label, cos_label, line_label)
        self.add(plot, labels)
```

Example 10: ArgMinExample - Interactive optimization visualization
```python
from manim import *

class ArgMinExample(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 10], y_range=[0, 100, 10], axis_config={"include_tip": False}
        )
        labels = ax.get_axis_labels(x_label="x", y_label="f(x)")

        t = ValueTracker(0)

        def func(x):
            return 2 * (x - 5) ** 2
        graph = ax.plot(func, color=MAROON)

        initial_point = [ax.coords_to_point(t.get_value(), func(t.get_value()))]
        dot = Dot(point=initial_point)

        dot.add_updater(lambda x: x.move_to(ax.c2p(t.get_value(), func(t.get_value()))))
        x_space = np.linspace(*ax.x_range[:2],200)
        minimum_index = func(x_space).argmin()

        self.add(ax, labels, graph, dot)
        self.play(t.animate.set_value(x_space[minimum_index]))
        self.wait()
```

Example 11: GraphAreaPlot - Area under curves and Riemann rectangles
```python
from manim import *

class GraphAreaPlot(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 5],
            y_range=[0, 6],
            x_axis_config={"numbers_to_include": [2, 3]},
            tips=False,
        )

        labels = ax.get_axis_labels()

        curve_1 = ax.plot(lambda x: 4 * x - x ** 2, x_range=[0, 4], color=BLUE_C)
        curve_2 = ax.plot(
            lambda x: 0.8 * x ** 2 - 3 * x + 4,
            x_range=[0, 4],
            color=GREEN_B,
        )

        line_1 = ax.get_vertical_line(ax.input_to_graph_point(2, curve_1), color=YELLOW)
        line_2 = ax.get_vertical_line(ax.i2gp(3, curve_1), color=YELLOW)

        riemann_area = ax.get_riemann_rectangles(curve_1, x_range=[0.3, 0.6], dx=0.03, color=BLUE, fill_opacity=0.5)
        area = ax.get_area(curve_2, [2, 3], bounded_graph=curve_1, color=GREY, opacity=0.5)

        self.add(ax, labels, curve_1, curve_2, line_1, line_2, riemann_area, area)
```

Example 12: PolygonOnAxes - Dynamic polygon areas with value tracking
```python
from manim import *

class PolygonOnAxes(Scene):
    def get_rectangle_corners(self, bottom_left, top_right):
        return [
            (top_right[0], top_right[1]),
            (bottom_left[0], top_right[1]),
            (bottom_left[0], bottom_left[1]),
            (top_right[0], bottom_left[1]),
        ]

    def construct(self):
        ax = Axes(
            x_range=[0, 10],
            y_range=[0, 10],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": False},
        )

        t = ValueTracker(5)
        k = 25

        graph = ax.plot(
            lambda x: k / x,
            color=YELLOW_D,
            x_range=[k / 10, 10.0, 0.01],
            use_smoothing=False,
        )

        def get_rectangle():
            polygon = Polygon(
                *[
                    ax.c2p(*i)
                    for i in self.get_rectangle_corners(
                        (0, 0), (t.get_value(), k / t.get_value())
                    )
                ]
            )
            polygon.stroke_width = 1
            polygon.set_fill(BLUE, opacity=0.5)
            polygon.set_stroke(YELLOW_B)
            return polygon

        polygon = always_redraw(get_rectangle)

        dot = Dot()
        dot.add_updater(lambda x: x.move_to(ax.c2p(t.get_value(), k / t.get_value())))
        dot.set_z_index(10)

        self.add(ax, graph, dot)
        self.play(Create(polygon))
        self.play(t.animate.set_value(10))
        self.play(t.animate.set_value(k / 10))
        self.play(t.animate.set_value(5))
```

Example 13: HeatDiagramPlot - Scientific data visualization
```python
from manim import *

class HeatDiagramPlot(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 40, 5],
            y_range=[-8, 32, 5],
            x_length=9,
            y_length=6,
            x_axis_config={"numbers_to_include": np.arange(0, 40, 5)},
            y_axis_config={"numbers_to_include": np.arange(-5, 34, 5)},
            tips=False,
        )
        labels = ax.get_axis_labels(
            x_label=Tex(r"$\Delta Q$"), y_label=Tex(r"T[$^\circ C$]")
        )

        x_vals = [0, 8, 38, 39]
        y_vals = [20, 0, 0, -5]
        graph = ax.plot_line_graph(x_values=x_vals, y_values=y_vals)

        self.add(ax, labels, graph)
```

Example 14: FollowingGraphCamera - Advanced camera movements
```python
from manim import *

class FollowingGraphCamera(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()

        # create the axes and the curve
        ax = Axes(x_range=[-1, 10], y_range=[-1, 10])
        graph = ax.plot(lambda x: np.sin(x), color=BLUE, x_range=[0, 3 * PI])

        # create dots based on the graph
        moving_dot = Dot(ax.i2gp(graph.t_min, graph), color=ORANGE)
        dot_1 = Dot(ax.i2gp(graph.t_min, graph))
        dot_2 = Dot(ax.i2gp(graph.t_max, graph))

        self.add(ax, graph, dot_1, dot_2, moving_dot)
        self.play(self.camera.frame.animate.scale(0.5).move_to(moving_dot))

        def update_curve(mob):
            mob.move_to(moving_dot.get_center())

        self.camera.frame.add_updater(update_curve)
        self.play(MoveAlongPath(moving_dot, graph, rate_func=linear))
        self.camera.frame.remove_updater(update_curve)

        self.play(Restore(self.camera.frame))
```

Example 15: ThreeDSurfacePlot - 3D mathematical surfaces
```python
from manim import *

class ThreeDSurfacePlot(ThreeDScene):
    def construct(self):
        resolution_fa = 24
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        def param_gauss(u, v):
            x = u
            y = v
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])

        gauss_plane = Surface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-2, +2],
            u_range=[-2, +2]
        )

        gauss_plane.scale(2, about_point=ORIGIN)
        gauss_plane.set_style(fill_opacity=1,stroke_color=GREEN)
        gauss_plane.set_fill_by_checkerboard(ORANGE, BLUE, opacity=0.5)
        axes = ThreeDAxes()
        self.add(axes,gauss_plane)
```

🎓 KEY PATTERNS FROM EXAMPLES:
- Use ValueTracker() for dynamic values that change over time
- Implement .add_updater() for objects that need to update automatically
- Use always_redraw() for objects that need constant redrawing
- Combine VGroup() to manage multiple related objects
- Apply .animate for smooth transformations
- Use proper positioning with .next_to(), .move_to(), .shift()
- Create custom functions for complex mathematical visualizations
- Use axes.plot() for mathematical functions and axes.get_area() for regions
- Implement SurroundingRectangle() for highlighting elements
- Use MathTex() for mathematical expressions and Text() for regular text
- Apply proper color schemes and opacity for visual clarity
- Use .set_z_index() to control layering of objects

EDUCATIONAL STEPS TO IMPLEMENT:"""]
        
        # Add detailed information about each educational step
        for i, step in enumerate(steps, 1):
            step_title = step.get('step_title', 'Step {}'.format(i))
            prompt_parts.append("""

Step {step_num}: {step_title}
- Duration: {duration} seconds
- Key Concepts: {key_concepts}
- Narration: {narration}
- Visual Plan: {visual_plan}
- Visual Elements: {visual_elements}
- Equations: {equations}
- Real-world Examples: {examples}""".format(
                step_num=i,
                step_title=step_title,
                duration=step.get('duration_seconds', 30),
                key_concepts=', '.join(step.get('key_concepts', [])),
                narration=step.get('narration_script', ''),
                visual_plan=step.get('animation_plan', ''),
                visual_elements=step.get('visual_elements', {}),
                equations=step.get('equations', []),
                examples=step.get('real_world_examples', [])
            ))

        # Create class name safely outside f-string
        class_name = title.replace(' ', '').replace(':', '').replace('(', '').replace(')', '').replace('-', '').replace("'", "").replace('"', '')
        if not class_name:
            class_name = "Educational"
        
        complexity = educational_breakdown.get('metadata', {}).get('difficulty_progression', 'intermediate')
        prompt_parts.append("""

TOTAL DURATION: {duration} seconds
TARGET COMPLEXITY: {complexity}

OUTPUT FORMAT:
Provide complete, executable Manim Python code following this structure:

```python
from manim import *

class {class_name}Scene(Scene):
    def construct(self):""".format(duration=duration, complexity=complexity, class_name=class_name) + """
        # Main orchestration method - CLEAR between each step
        self.intro_sequence()
        self.clear_and_transition()
        self.step_1_introduction()
        self.clear_and_transition()
        self.step_2_core_concepts()
        self.clear_and_transition()
        # ... more steps as needed
        self.conclusion_summary()
    
    def clear_and_transition(self):
        # Clean transition between sections
        self.play(FadeOut(*self.mobjects))
        self.wait(0.5)
    
    def intro_sequence(self):
        # Engaging introduction with DYNAMIC positioning
        title = Text("{{title}}", font_size=48, color=BLUE).shift(UP*3)
        subtitle = Text("Educational Animation", font_size=32, color=WHITE).shift(UP*1.5)
        
        self.play(Write(title))
        self.wait(0.5)
        self.play(Write(subtitle))
        self.wait(1)
        
        # Move content and add more
        self.play(
            title.animate.shift(LEFT*3).scale(0.7),
            subtitle.animate.shift(RIGHT*3).scale(0.8)
        )
        
        intro_text = Text("Let's explore this concept step by step", 
                          font_size=24, color=YELLOW).shift(DOWN*1)
        self.play(FadeIn(intro_text))
        self.wait(2)
        
    def step_1_introduction(self):
        # First educational step - NEW positions, no overlap
        step_title = Text("Step 1: Foundation", font_size=40, color=BLUE).shift(UP*2.5)
        self.play(Write(step_title))
        
        # Create diagram on LEFT, text on RIGHT
        diagram = Circle(radius=1, color=WHITE).shift(LEFT*3)
        explanation = Text("Key concept explanation\\nwith multiple lines", 
                          font_size=20, color=WHITE).shift(RIGHT*2)
        
        self.play(Create(diagram), Write(explanation))
        self.wait(1)
        
        # Transform and move
        new_shape = Square(side_length=2, color=YELLOW).shift(LEFT*3)
        self.play(Transform(diagram, new_shape))
        
        # Add connecting arrow
        arrow = Arrow(LEFT*1, RIGHT*0.5, color=GREEN)
        self.play(Create(arrow))
        self.wait(2)
        
    # CONTINUE with similar patterns for each step...
```

CRITICAL REQUIREMENTS:
1. Generate COMPLETE, EXECUTABLE code
2. Include ALL necessary imports
3. Follow proper Manim syntax and conventions
4. Create visually appealing, educational animations
5. Ensure smooth flow between all steps
6. Use dynamic positioning - avoid static layouts
7. Include proper documentation and comments
8. Make the code modular and easy to understand
9. Optimize for visual clarity and educational impact
10. Follow the educational step progression exactly
11. Must Contain all necessary imports and class definitions
12. def construct() method must orchestrate the entire scene flow
13. ⚠️ NEVER use ImageMobject or image file references ⚠️
14. Use only built-in Manim objects (Text, MathTex, shapes, etc.)
15. Create visual representations using geometric shapes and text
16. Represent real-world examples with descriptive text and shape-based diagrams
17. ANIMATE EVERYTHING - no static content allowed
18. Use self.clear() or FadeOut(*self.mobjects) between major sections
19. Move objects around the screen dynamically with .animate.shift()
20. Transform objects instead of creating new ones in same position
21. Use proper spacing - NEVER overlap text at same coordinates
22. Implement smooth transitions between concepts
23. Create engaging visual flow with object movements
24. ONLY use valid Scene methods: self.add(), self.play(), self.wait(), self.clear(), self.remove()
25. NEVER use self.set_background() or similar invalid methods
26. ALWAYS position objects at different coordinates using UP*2, DOWN*1, LEFT*3, RIGHT*2
27. Clear screen between sections: self.play(FadeOut(*self.mobjects))
28. Use different font sizes to create hierarchy: 48 for titles, 36 for subtitles, 24 for content

⚠️ CRITICAL SYNTAX REQUIREMENTS ⚠️:
- NEVER write Text("text",.shift() - comma before method is SYNTAX ERROR
- ALWAYS write Text("text").shift() - proper method chaining
- NEVER write Text("text").shift(UP*2 - missing closing parenthesis is SYNTAX ERROR  
- ALWAYS write Text("text").shift(UP*2) - complete parentheses
- NEVER split Text declarations across multiple lines
- ALWAYS complete Text objects on single lines
- NEVER create orphaned lines starting with font_size= or color=
- ALWAYS use proper 4-space indentation for class methods

ANIMATION REQUIREMENTS:
- Every text element should be animated (Write, FadeIn, etc.)
- Use Transform() to morph objects between states
- Implement smooth camera movements when appropriate
- Clear previous content before introducing new concepts: self.play(FadeOut(*self.mobjects))
- Position elements strategically using UP, DOWN, LEFT, RIGHT with multipliers
- Use scale and rotation for visual interest: .scale(0.8), .rotate(PI/4)
- Implement highlighting effects (Indicate, Flash, Wiggle)
- Create progressive reveals for complex concepts
- Use color changes to show relationships: .set_color(BLUE)
- Implement step-by-step builds for equations and diagrams

MANDATORY POSITIONING EXAMPLES:
- Title: Text("Title", font_size=48).shift(UP*3)
- Subtitle: Text("Subtitle", font_size=36).shift(UP*1.5)  
- Content: Text("Content", font_size=24).shift(DOWN*1)
- Left diagram: Circle().shift(LEFT*4)
- Right text: Text("Explanation").shift(RIGHT*3)
- Multiple items: use UP*2, CENTER, DOWN*2 for vertical spacing
- NEVER put two Text objects in the same position
- ALWAYS move or remove old content before adding new content

FORBIDDEN ELEMENTS:
- ImageMobject (will cause file not found errors)
- Any references to .png, .jpg, .jpeg, .gif files
- External image assets
- File loading operations
- self.set_background() method (AttributeError - doesn't exist)
- self.set_color_scheme() method (AttributeError - not valid)
- self.set_theme() method (AttributeError - not valid)
- self.configure_camera() method (AttributeError - not a Scene method)

VALID SCENE METHODS TO USE:
- self.add() - add objects to scene
- self.play() - animate objects
- self.wait() - pause between animations
- self.clear() - clear all objects from scene
- self.remove() - remove specific objects
- self.camera - access camera properties (read-only)

USE INSTEAD:
- Text() for descriptions and labels
- MathTex() for mathematical expressions
- Circle(), Rectangle(), Line() for diagrams
- Color-coded shapes to represent concepts
- Animated text reveals and transformations


Generate the complete Manim code now. Ensure it's production-ready and follows all the requirements above.
""")
        
        return ''.join(prompt_parts)

    def _extract_manim_code(self, response):
        """
        Extract Manim code from AI response with multiple strategies.
        
        Args:
            response (str): AI response containing Manim code
            
        Returns:
            str: Extracted Manim code or None
        """
        # Strategy 1: Look for python code blocks
        python_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if python_blocks:
            code = python_blocks[0].strip()
            if 'from manim import' in code or 'import manim' in code:
                return code
        
        # Strategy 2: Look for any code blocks
        code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
        for block in code_blocks:
            if 'from manim import' in block or 'class' in block and 'Scene' in block:
                return block.strip()
        
        # Strategy 3: Look for class definitions in the text
        class_pattern = r'(class.*?Scene.*?:\s*.*?(?=class|$))'
        class_matches = re.findall(class_pattern, response, re.DOTALL)
        if class_matches:
            code = class_matches[0].strip()
            # Add imports if missing
            if 'from manim import' not in code:
                code = 'from manim import *\n\n' + code
            return code
        
        # Strategy 4: Extract everything that looks like Python code
        if 'def construct' in response:
            # Try to extract the main code portion
            lines = response.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if 'from manim import' in line or 'class' in line:
                    in_code = True
                
                if in_code:
                    code_lines.append(line)
                    
                # Stop at certain markers
                if line.strip() == '' and in_code and len(code_lines) > 20:
                    break
            
            if code_lines:
                return '\n'.join(code_lines)
        
        return None

    def _create_manim_generation_prompt(self):
        """
        Create the system prompt for Manim code generation.
        
        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        system_message = SystemMessage(
            content='''
            You are an expert Manim (Mathematical Animation Engine) code generator, 
            specializing in creating educational animations in the style of 3Blue1Brown.
            
            Your expertise includes:
            - Converting educational content into dynamic, visual animations
            - Creating smooth transitions and engaging reveals
            - Implementing proper mathematical notation and diagrams
            - Designing pedagogically effective visual sequences
            - Following Manim best practices and conventions
            - Generating clean, modular, and well-documented code
            
            CRITICAL CONSTRAINTS:
            - NEVER use ImageMobject or any image file references
            - Use only built-in Manim objects (Text, MathTex, Circle, Rectangle, Line, etc.)
            - Create visual diagrams using geometric shapes and mathematical objects
            - Represent real-world examples with text descriptions and geometric visualizations
            - ANIMATE EVERYTHING - no static content is allowed
            - Use proper scene management with clear transitions
            - Implement dynamic positioning and object movements
            - Avoid overlapping text and cluttered layouts
            - DO NOT use invalid methods like self.set_background(), self.set_color_scheme(), etc.
            - Background colors are handled automatically by Manim or in config files
            - Use only valid Scene class methods like self.add(), self.play(), self.wait(), self.clear()
            - CRITICAL: Never write Text("text",.shift() - always use Text("text").shift()
            - CRITICAL: Method chaining must have proper syntax: .shift(UP*1).scale(0.8)
            - CRITICAL: No comma before dot operators: Text("text").shift() NOT Text("text",.shift()
            
            FORBIDDEN METHODS (will cause AttributeError):
            - self.set_background() - doesn't exist in Manim
            - self.set_color_scheme() - not a valid Scene method
            - self.set_theme() - not a valid Scene method
            - self.configure_camera() - not a Scene method
            
            FORBIDDEN SYNTAX PATTERNS (will cause SyntaxError):
            - Text("text",.shift(UP*1) ❌ (comma before method)
            - Text("text").shift(UP*2 ❌ (missing closing parenthesis)
            - Text("text",.shift(UP*2) ❌ (comma + missing closing paren)
            - Text("text", .shift(UP*2) ❌ (comma space before method)
            
            CORRECT SYNTAX EXAMPLES:
            - Text("Hello World").shift(UP*2) ✅
            - Text("Hello", font_size=24).shift(DOWN*1) ✅
            - Text("Title", color=BLUE).shift(UP*3).scale(0.8) ✅
            
            MANDATORY SYNTAX RULES:
            1. NEVER write Text("text",.shift() - always use Text("text").shift()
            2. NEVER write Text("text", .method() - always use Text("text").method()
            3. ALWAYS close parentheses: Text("text") NOT Text("text"
            4. ALWAYS use proper method chaining: .shift(UP*1).scale(0.8)
            5. NO orphaned lines starting with font_size= or color=
            6. Every Text object must be complete on ONE line
            7. Use proper indentation (4 spaces) for methods inside classes
            
            When generating Manim code, you MUST:
            1. Create complete, executable Python code
            2. Use proper Manim imports and syntax
            3. Implement dynamic positioning and smooth animations
            4. Clear scenes between major sections using self.play(FadeOut(*self.mobjects))
            5. Move objects around the screen - avoid static positioning
            6. Use Transform() to morph objects between states
            7. Position elements using UP, DOWN, LEFT, RIGHT vectors with multipliers (UP*2, LEFT*3)
            8. Include appropriate timing and transitions
            9. Follow the educational flow provided in the input
            10. Add comprehensive comments explaining the code
            11. Use engaging visual elements and color schemes
            12. Implement proper scene management and cleanup
            13. Create modular methods for different educational steps
            14. Ensure the code is production-ready and can be executed directly with Manim
            15. Use the provided educational breakdown to structure the code
            16. Generate code that is easy to modify and extend
            17. Use advanced Manim features like Write(), FadeIn(), Transform(), and Create()
            18. Ensure the code is well-structured and follows Python conventions
            19. Use dynamic camera movements and object transformations
            20. Include all necessary imports and class definitions
            21. Use clear, descriptive class and method names
            22. Ensure the construct() method orchestrates the entire scene flow
            23. NEVER reference external image files - use only text and geometric shapes
            24. Animate every text element with Write(), FadeIn(), or similar
            25. Use proper spacing and avoid overlapping content
            26. Create visual flow with object movements and morphing
            27. Implement step-by-step reveals for complex concepts
            28. Use highlighting effects like Indicate(), Flash(), Wiggle()
            29. Position objects strategically and move them dynamically
            30. Transform objects instead of creating static new ones
            31. CRITICAL: Never place multiple objects at the same coordinates
            32. CRITICAL: Always use different positions like UP*3, UP*1, CENTER, DOWN*1, DOWN*3
            33. CRITICAL: Clear previous content before adding new content
            34. CRITICAL: Use LEFT and RIGHT sides for diagrams vs text
            35. CRITICAL: Create visual hierarchy with different font sizes
            Always generate production-ready code that can be executed directly with Manim.
            '''
        )

        human_message_prompt = HumanMessagePromptTemplate.from_template("{human_input}")

        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            human_message_prompt,
        ])
        return prompt

    def _display_video_plan(self, video_plan):
        """
        Display video plan details in terminal for debugging and monitoring.
        
        Args:
            video_plan (dict): Complete video plan from script generator
        """
        print("\n" + "="*80)
        print("📋 VIDEO PLAN DETAILS")
        print("="*80)
        
        educational_breakdown = video_plan.get("educational_breakdown", {})
        manim_structure = video_plan.get("manim_structure", {})
        generation_metadata = video_plan.get("generation_metadata", {})
        
        # Basic info
        print("🎯 Title: {}".format(educational_breakdown.get('title', 'N/A')))
        print("📝 Abstract: {}...".format(educational_breakdown.get('abstract', 'N/A')[:100]))
        print("⏱️  Duration: {} seconds".format(educational_breakdown.get('metadata', {}).get('estimated_total_duration', 'N/A')))
        print("👥 Target Audience: {}".format(educational_breakdown.get('metadata', {}).get('target_audience', 'N/A')))
        
        # Learning objectives
        objectives = educational_breakdown.get('learning_objectives', [])
        if objectives:
            print("\n🎯 Learning Objectives ({}):".format(len(objectives)))
            for i, obj in enumerate(objectives[:3], 1):
                print("   {}. {}".format(i, obj))
            if len(objectives) > 3:
                print("   ... and {} more".format(len(objectives) - 3))
        
        # Educational steps
        steps = educational_breakdown.get('educational_steps', [])
        if steps:
            print("\n📚 Educational Steps ({}):".format(len(steps)))
            for i, step in enumerate(steps, 1):
                title = step.get('step_title', 'Step {}'.format(i))
                duration = step.get('duration_seconds', 'N/A')
                concepts = step.get('key_concepts', [])
                print("   {}. {} ({}s)".format(i, title, duration))
                if concepts:
                    print("      Key Concepts: {}".format(', '.join(concepts[:3])))
        
        # Manim structure
        if manim_structure:
            animation_steps = manim_structure.get('animation_steps', [])
            print("\n🎬 Animation Steps ({}):".format(len(animation_steps)))
            for i, step in enumerate(animation_steps[:3], 1):
                objects = step.get('manim_objects', [])
                animations = step.get('animations', [])
                print("   {}. {}".format(i, step.get('description', 'Animation step')))
                print("      Objects: {}".format(', '.join(objects[:3])))
                print("      Animations: {}".format(', '.join(animations[:3])))
            if len(animation_steps) > 3:
                print("   ... and {} more steps".format(len(animation_steps) - 3))
        
        # Generation metadata
        if generation_metadata:
            print("\n⚙️  Generation Info:")
            print("   Stages Completed: {}".format(generation_metadata.get('stages_completed', [])))
            print("   Total Duration: {} seconds".format(generation_metadata.get('total_duration', 'N/A')))
            print("   Complexity: {}".format(generation_metadata.get('complexity_level', 'N/A')))
        
        print("="*80)

    def _display_manim_code(self, manim_code):
        """
        Display generated Manim code in terminal with formatting.
        
        Args:
            manim_code (str): Generated Manim Python code
        """
        print("\n" + "="*80)
        print("🐍 GENERATED MANIM CODE")
        print("="*80)
        
        # Code analysis
        lines = manim_code.split('\n')
        print("📊 Code Statistics:")
        print("   Lines of code: {}".format(len(lines)))
        print("   Characters: {}".format(len(manim_code)))
        
        # Check for key components
        has_imports = 'from manim import' in manim_code or 'import manim' in manim_code
        has_class = 'class' in manim_code and 'Scene' in manim_code
        has_construct = 'def construct' in manim_code
        has_methods = manim_code.count('def ') > 1
        
        print("   Has imports: {}".format('✅' if has_imports else '❌'))
        print("   Has scene class: {}".format('✅' if has_class else '❌'))
        print("   Has construct method: {}".format('✅' if has_construct else '❌'))
        print("   Has additional methods: {}".format('✅' if has_methods else '❌'))
        
        # Extract class name
        import re
        class_match = re.search(r'class\s+(\w+)', manim_code)
        if class_match:
            print("   Class name: {}".format(class_match.group(1)))
        
        # Show first few lines and last few lines
        print("\n📝 Code Preview:")
        print("─" * 40)
        
        # First 15 lines
        for i, line in enumerate(lines[:15], 1):
            print("{:2}: {}".format(i, line))
        
        if len(lines) > 30:
            print("   ...")
            print("   [... {} lines omitted ...]".format(len(lines) - 30))
            print("   ...")
            
            # Last 15 lines
            for i, line in enumerate(lines[-15:], len(lines) - 14):
                print("{:2}: {}".format(i, line))
        elif len(lines) > 15:
            # Show remaining lines if total is between 15-30
            for i, line in enumerate(lines[15:], 16):
                print("{:2}: {}".format(i, line))
        
        print("─" * 40)
        print("🎬 Code ready for Manim rendering!")
        print("="*80)

    def _validate_and_fix_manim_code(self, code):
        """
        Validate and fix common issues in generated Manim code.
        
        Args:
            code (str): Generated Manim code
            
        Returns:
            str: Fixed and validated Manim code
        """
        # Normalize indentation to avoid unexpected indent errors
        code = textwrap.dedent(code)
        print("🔧 Starting code validation and fixing...")
        
        # CRITICAL: Fix syntax errors FIRST before anything else
        code = self._fix_syntax_errors(code)
        print("✅ Syntax errors fixed")
        
        lines = code.split('\n')
        fixed_lines = []
        text_positions_used = set()
        
        for line in lines:
            # Fix set_background method which doesn't exist in Manim
            if 'self.set_background' in line:
                # Comment out the problematic line and add explanation
                fixed_lines.append("        # REMOVED: {} (set_background method doesn't exist in Manim)".format(line.strip()))
                fixed_lines.append("        # Background color is set in Manim config or using Camera background_color")
                fixed_lines.append("        # Scene backgrounds are handled automatically by Manim")
                continue
            
            # Fix overlapping text positions - detect Text objects without positioning
            if 'Text(' in line and '=' in line and 'shift(' not in line and 'move_to(' not in line:
                # This is a Text object without explicit positioning - add positioning
                if 'title' in line.lower():
                    line = line.rstrip() + '.shift(UP*3)'
                elif 'subtitle' in line.lower():
                    line = line.rstrip() + '.shift(UP*1.5)'
                elif 'step' in line.lower() and 'title' in line.lower():
                    line = line.rstrip() + '.shift(UP*2)'
                else:
                    # Add random positioning to avoid overlap
                    positions = ['UP*1', 'DOWN*1', 'LEFT*2', 'RIGHT*2', 'UP*2', 'DOWN*2']
                    for pos in positions:
                        if pos not in text_positions_used:
                            line = line.rstrip() + '.shift({})'.format(pos)
                            text_positions_used.add(pos)
                            break
            
            # Check for missing scene clearing between methods
            if 'def ' in line and 'construct' not in line and '__init__' not in line:
                # This is a new method - ensure it starts with clearing if it's not the first method
                method_name = line.strip()
                fixed_lines.append(line)
                # Add clearing instruction as comment
                fixed_lines.append("        # Clear previous content to avoid overlap")
                fixed_lines.append("        # self.play(FadeOut(*self.mobjects)) # Uncomment if needed")
                continue
            
            # Skip lines with ImageMobject references
            if 'ImageMobject' in line or 'Image.open' in line:
                # Replace with a comment explaining what was removed
                comment_line = line.strip()
                fixed_lines.append("        # REMOVED: {} (ImageMobject not supported)".format(comment_line))
                fixed_lines.append("        # Using text description instead:")
                
                # Extract variable name if possible
                if '=' in line and 'ImageMobject' in line:
                    var_name = line.split('=')[0].strip()
                    # Replace with a text description with positioning
                    fixed_lines.append("        {} = Text('Visual representation of concept', font_size=24).shift(DOWN*1)".format(var_name))
                continue
            
            # Skip lines that reference image files
            if any(ext in line.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.ico']):
                # Comment out the problematic line
                fixed_lines.append("        # REMOVED: {} (Image file reference not supported)".format(line.strip()))
                continue
            
            # Fix other common Manim method errors
            if any(method in line for method in ['self.set_color_scheme', 'self.set_theme', 'self.configure_camera']):
                # Comment out invalid methods
                fixed_lines.append("        # REMOVED: {} (Invalid Manim method)".format(line.strip()))
                continue
                
            # Fix common import issues
            if line.strip() == 'from manim import *':
                fixed_lines.append(line)
            elif 'import' in line and any(img_ref in line for img_ref in ['PIL', 'Image', 'cv2', 'opencv']):
                # Comment out image-related imports
                fixed_lines.append("        # REMOVED: {} (Image library import not needed)".format(line.strip()))
                continue
            else:
                fixed_lines.append(line)
        
        # Join the fixed lines
        fixed_code = '\n'.join(fixed_lines)
        
        # Apply syntax fixes AGAIN after other modifications
        fixed_code = self._fix_syntax_errors(fixed_code)
        print("✅ Applied syntax fixes again after other modifications")
        
        # Handle empty method bodies after removing invalid lines
        # If a method becomes empty (only has pass or comments), add a pass statement
        import re
        
        # Find method definitions and check if they're empty
        method_pattern = r'(def\s+\w+\([^)]*\):\s*)((?:\s*#.*\n)*)\s*$'
        def fix_empty_method(match):
            method_def = match.group(1)
            comments = match.group(2) if match.group(2) else ""
            # If there are only comments or nothing after the method definition, add pass
            if not comments.strip() or all(line.strip().startswith('#') or not line.strip() for line in comments.split('\n')):
                return method_def + '\n' + comments + '        pass\n'
            return match.group(0)
        
        fixed_code = re.sub(method_pattern, fix_empty_method, fixed_code, flags=re.MULTILINE)
        
        # Ensure we have proper imports
        if 'from manim import *' not in fixed_code:
            fixed_code = 'from manim import *\n\n' + fixed_code
        
        # Final syntax validation
        try:
            import ast
            ast.parse(fixed_code)
            print("✅ Final syntax validation passed")
        except SyntaxError as e:
            print("❌ Final syntax error detected: {}".format(e.msg))
            print("   Problem line {}: {}".format(e.lineno, e.text))
            # Try one more aggressive fix
            fixed_code = self._emergency_syntax_fix(fixed_code, e)
        
        return fixed_code

    def _fix_syntax_errors(self, code):
        """
        Fix common syntax errors in generated Manim code.
        
        Args:
            code (str): Original Manim code
            
        Returns:
            str: Code with syntax errors fixed
        """
        if not code:
            return code
        
        import re
        import ast
        
        print("🔧 Starting comprehensive syntax error fixing...")
        
        # Step 1: Fix orphaned lines and incomplete Text declarations
        lines = code.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                i += 1
                continue
            
            # Detect orphaned method calls (lines that start with parameters or method calls)
            if stripped.startswith(('font_size=', 'color=', '.shift(', '.scale(', '.rotate(')):
                print("🔧 Removing orphaned line {}: {}".format(i+1, stripped))
                # Skip this line entirely as it's a broken continuation
                i += 1
                continue
            
            # Fix the main problematic pattern: Text("text",.shift(
            if 'Text(' in line and '",' in line and any(method in line for method in ['.shift(', '.scale(', '.rotate(']):
                # Pattern: Text("some text",.shift(UP*1) -> Text("some text").shift(UP*1)
                original_line = line
                # Fix comma before method calls
                line = re.sub(r'Text\("([^"]*)",\s*\.(shift|scale|rotate|set_color)\(', r'Text("\1").\2(', line)
                if line != original_line:
                    print("🔧 Fixed Text+comma+method pattern: {}".format(line.strip()))
            
            # Fix pattern: Text("text"), .method( -> Text("text").method(
            if 'Text(' in line and '"), .' in line:
                original_line = line
                line = re.sub(r'Text\("([^"]*)"\),\s*\.(shift|scale|rotate|set_color)\(', r'Text("\1").\2(', line)
                if line != original_line:
                    print("🔧 Fixed Text+closing+comma+method pattern: {}".format(line.strip()))
            
            # Fix incomplete Text declarations that may span multiple lines
            if 'Text(' in line and not ')' in line and not line.strip().endswith(','):
                # This might be an incomplete Text declaration - look ahead
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith(('font_size=', 'color=')) and ')' in next_line:
                        # Combine the lines properly
                        combined = line.rstrip() + ', ' + next_line
                        # Ensure proper syntax
                        if combined.count('(') == combined.count(')'):
                            fixed_lines.append(combined)
                            print("🔧 Combined incomplete Text declaration: {}".format(combined.strip()))
                            i += 2  # Skip both lines
                            continue
            
            # Handle missing closing parentheses
            if 'Text(' in line and line.count('(') > line.count(')'):
                missing_parens = line.count('(') - line.count(')')
                original_line = line
                line = line.rstrip() + ')' * missing_parens
                if line != original_line:
                    print("🔧 Added {} missing closing parenthesis(es): {}".format(missing_parens, line.strip()))
            
            fixed_lines.append(line)
            i += 1
        
        # Rebuild the code
        code = '\n'.join(fixed_lines)
        
        # Step 2: Apply regex-based fixes
        print("🔧 Applying regex-based syntax fixes...")
        
        # Fix patterns in order of complexity
        patterns_and_fixes = [
            # Fix Text("text",.shift() patterns
            (r'Text\("([^"]*)",\s*\.shift\(', r'Text("\1").shift('),
            (r'Text\("([^"]*)",\s*\.scale\(', r'Text("\1").scale('),
            (r'Text\("([^"]*)",\s*\.rotate\(', r'Text("\1").rotate('),
            
            # Fix trailing commas before method calls
            (r'",\s*\.(shift|scale|rotate|set_color)\(', r'").\1('),
            
            # Fix double commas
            (r',,+', ','),
            
            # Fix spaces around operators
            (r'\s+\.', '.'),
            
            # Fix missing closing parentheses at end of lines
            (r'Text\([^)]*$', lambda m: m.group(0) + ')'),
        ]
        
        for pattern, replacement in patterns_and_fixes:
            if callable(replacement):
                code = re.sub(pattern, replacement, code)
            else:
                code = re.sub(pattern, replacement, code)
        
        # Step 3: Final syntax validation and targeted fixes
        try:
            ast.parse(code)
            print("✅ Syntax validation passed")
        except SyntaxError as e:
            print("⚠️ Syntax error detected at line {}: {}".format(e.lineno, e.msg))
            
            if e.lineno and e.text:
                lines = code.split('\n')
                if 0 < e.lineno <= len(lines):
                    problem_line_index = e.lineno - 1
                    problem_line = lines[problem_line_index]
                    
                    print("   Problem line: {}".format(problem_line.strip()))
                    
                    # Apply targeted fixes based on error type
                    if isinstance(e, IndentationError):
                        if "unexpected indent" in e.msg.lower():
                            # Remove unexpected indentation
                            fixed_line = problem_line.lstrip()
                            lines[problem_line_index] = fixed_line
                            print("🔧 Removed unexpected indentation: {}".format(fixed_line.strip()))
                        elif "expected an indented block" in e.msg.lower():
                            # Add proper indentation
                            fixed_line = "    " + problem_line
                            lines[problem_line_index] = fixed_line
                            print("🔧 Added expected indentation: {}".format(fixed_line.strip()))
                    
                    elif 'invalid syntax' in e.msg.lower():
                        # Handle specific syntax issues
                        if problem_line.count('(') > problem_line.count(')'):
                            # Missing closing parentheses
                            missing = problem_line.count('(') - problem_line.count(')')
                            fixed_line = problem_line + ')' * missing
                            lines[problem_line_index] = fixed_line
                            print("🔧 Added {} missing closing parentheses".format(missing))
                        
                        elif ',.' in problem_line:
                            # Fix comma-dot patterns
                            fixed_line = problem_line.replace(',.', '.')
                            lines[problem_line_index] = fixed_line
                            print("🔧 Fixed comma-dot pattern: {}".format(fixed_line.strip()))
                    
                    # Rebuild and try again
                    code = '\n'.join(lines)
                    
                    try:
                        ast.parse(code)
                        print("✅ Syntax error fixed successfully")
                    except SyntaxError as e2:
                        print("❌ Could not fix syntax error: {}".format(e2.msg))
                        # Add error comment but keep the code
                        error_comment = "# SYNTAX ERROR: {} at line {}\n".format(e.msg, e.lineno)
                        code = error_comment + code
            else:
                print("❌ Syntax error without line information: {}".format(e.msg))
                error_comment = "# SYNTAX ERROR: {}\n".format(e.msg)
                code = error_comment + code
        
        return code

    def _emergency_syntax_fix(self, code, syntax_error):
        """
        Emergency fix for syntax errors that couldn't be resolved by normal means.
        
        Args:
            code (str): Code with syntax error
            syntax_error (SyntaxError): The syntax error object
            
        Returns:
            str: Code with emergency fixes applied
        """
        if not syntax_error.lineno or not syntax_error.text:
            return code
        
        lines = code.split('\n')
        if syntax_error.lineno > len(lines):
            return code
        
        problem_line = lines[syntax_error.lineno - 1]
        print("🚨 Emergency syntax fix for line {}: {}".format(syntax_error.lineno, problem_line.strip()))
        
        # Common emergency fixes
        fixed_line = problem_line
        
        # Fix the specific pattern we keep seeing
        if 'Text(' in problem_line and '",' in problem_line and '.shift(' in problem_line:
            # Pattern: Text("text",.shift(UP*1) -> Text("text").shift(UP*1)
            fixed_line = re.sub(r'Text\("([^"]*)",\s*\.shift\(([^)]*)\)', r'Text("\1").shift(\2)', problem_line)
            print("🔧 Emergency fix applied: {}".format(fixed_line.strip()))
        
        # Fix missing closing parentheses
        elif problem_line.count('(') > problem_line.count(')'):
            missing = problem_line.count('(') - problem_line.count(')')
            fixed_line = problem_line.rstrip() + ')' * missing
            print("🔧 Emergency fix: added {} closing parentheses".format(missing))
        
        # Fix missing opening parentheses (rare)
        elif problem_line.count(')') > problem_line.count('('):
            # Find where to add opening parentheses (usually after =)
            if '=' in problem_line and 'Text(' in problem_line:
                parts = problem_line.split('=', 1)
                if len(parts) == 2:
                    fixed_line = parts[0] + '= Text(' + parts[1].lstrip().lstrip('Text(')
                    print("🔧 Emergency fix: added opening parenthesis")
        
        # Fix malformed quotes
        elif problem_line.count('"') % 2 == 1:
            # Odd number of quotes - add one at the end of the string content
            if 'Text(' in problem_line:
                # Find the last quote and add a closing one
                last_quote_pos = problem_line.rfind('"')
                if last_quote_pos > 0 and last_quote_pos < len(problem_line) - 1:
                    # Check if we need to add quote before comma or other punctuation
                    next_char_pos = last_quote_pos + 1
                    if next_char_pos < len(problem_line) and problem_line[next_char_pos] in ',.):':
                        fixed_line = problem_line[:last_quote_pos + 1] + '"' + problem_line[next_char_pos:]
                        print("🔧 Emergency fix: added missing quote")
        
        # Apply the fix if we made one
        if fixed_line != problem_line:
            lines[syntax_error.lineno - 1] = fixed_line
            fixed_code = '\n'.join(lines)
            
            # Test if the fix worked
            try:
                import ast
                ast.parse(fixed_code)
                print("✅ Emergency fix successful!")
                return fixed_code
            except SyntaxError as e2:
                print("❌ Emergency fix failed: {}".format(e2.msg))
                # Return original code with a comment about the error
                return "# SYNTAX ERROR DETECTED: {}\n# LINE {}: {}\n\n".format(syntax_error.msg, syntax_error.lineno, syntax_error.text) + code
        
        # If no fix was applied, return original with error comment
        return "# SYNTAX ERROR DETECTED: {}\n# LINE {}: {}\n\n".format(syntax_error.msg, syntax_error.lineno, syntax_error.text) + code

# Initialize the Manim code generator
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    manim_generator = ManIMCodeGenerator(GOOGLE_API_KEY)
    print("✅ Advanced Manim Code Generator initialized successfully!")
    print("🎨 Ready to generate dynamic, educational animations with Google Gemini!")
else:
    print("❌ GOOGLE_API_KEY not found. Please set your API key in the .env file.")