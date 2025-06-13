import os
from dotenv import load_dotenv
import logging
import json
import re
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logging.getLogger('comtypes').setLevel(logging.WARNING)

load_dotenv('.env')

class Enhanced3B1BManimGenerator:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)  # Reduced memory
        self.groq_chat = ChatGroq(groq_api_key=self.groq_api_key, model_name='llama3-8b-8192')  # Smaller model
        self.prompt = self._create_enhanced_3b1b_prompt()
        self.conversation = ConversationChain(
            llm=self.groq_chat,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory,
            input_key="human_input",
        )
        self.generated_hashes = set()  # Track generated content to avoid repetition

    def generate_3b1b_manim_code(self, structured_content):
        """
        Generate 3Blue1Brown style Manim code with consistent structure and animations.
        """
        if not structured_content:
            return ""

        try:
            title = structured_content.get('title', 'Math Concept')
            domain = structured_content.get('mathematical_domain', 'general')
            
            # Check for repetition
            content_hash = hash(str(structured_content))
            if content_hash in self.generated_hashes:
                print("Similar content detected, adding variation...")
                structured_content['_variation'] = len(self.generated_hashes)
            
            self.generated_hashes.add(content_hash)
            
            # Create a concise prompt
            manim_prompt = self._create_structured_animation_prompt(title, domain, [], structured_content)
            
            response = self.conversation.predict(human_input=manim_prompt)
            cleaned_code = self._clean_and_enhance_code(response)
            
            if not cleaned_code:
                print("LLM generated invalid code, using enhanced fallback...")
                return self._generate_enhanced_3b1b_fallback(structured_content)
            
            return cleaned_code

        except Exception as e:
            print(f"Error generating 3B1B Manim code: {e}")
            return self._generate_enhanced_3b1b_fallback(structured_content)

    def _create_structured_animation_prompt(self, title, domain, steps, content):
        """Create a concise, varied prompt for consistent animations."""
        # Clean title for class name
        clean_title = re.sub(r'[^a-zA-Z0-9]', '', title.replace(' ', ''))
        if not clean_title:
            clean_title = "MathAnimation"
        
        # Generate a unique seed based on content to ensure variety
        content_hash = hash(str(content)) % 1000
        
        # Get domain-specific requirements (shortened)
        domain_specs = self._get_compact_domain_specs(domain, content_hash)
        
        # Create varied approach based on content hash
        approach_variants = [
            "animated construction", "dynamic transformation", "step-by-step revelation",
            "geometric interpretation", "algebraic visualization", "conceptual building"
        ]
        chosen_approach = approach_variants[content_hash % len(approach_variants)]
        
        return f"""Create a unique {domain} Manim animation using {chosen_approach} approach.

SPECIFICATIONS:
- Title: {title}
- Class: {clean_title}Scene  
- Style: 3Blue1Brown (BLUE_E, TEAL, YELLOW, RED_B, GREEN_B, GOLD)
- Structure: 5 methods (setup_scene, introduce_concept, animate_main_content, reveal_key_insight, conclude_animation)

DOMAIN FOCUS ({domain}):
{domain_specs}

UNIQUENESS REQUIREMENT #{content_hash}:
Make this animation visually distinct from previous generations. Use creative mathematical visualizations, unique color combinations from the 3B1B palette, and innovative animation sequences.

CRITICAL FIXES:
- DO NOT override text elements - use FadeOut on old elements before creating new ones
- Use self.remove() to clear objects that might conflict
- Position elements carefully to avoid overlap
- Each animation should be mathematically different and visually unique

Generate ONLY complete Python code with proper object management and no text overriding."""

    def _get_compact_domain_specs(self, domain, variant_seed):
        """Get compact, varied domain specifications."""
        
        # Create variations based on seed
        colors = [['BLUE_E', 'TEAL'], ['YELLOW', 'GREEN_B'], ['RED_B', 'GOLD']][variant_seed % 3]
        primary_color, secondary_color = colors
        
        specs = {
            "calculus": f"""
- Show function f(x) with derivative visualization
- Use {primary_color} for functions, {secondary_color} for tangents
- Animate rate of change concepts
- Include mathematical equations with MathTex
- Use NumberPlane coordinate system
            """,
            "algebra": f"""
- Demonstrate equation solving steps
- Use {primary_color} for equations, {secondary_color} for solutions  
- Show algebraic transformations
- Include geometric interpretation if possible
- Animate ReplacementTransform for equation changes
            """,
            "geometry": f"""
- Construct geometric shapes step-by-step
- Use {primary_color} for shapes, {secondary_color} for measurements
- Show geometric relationships and theorems
- Include angle measurements and side lengths
- Use proper geometric objects (Circle, Polygon, Line, Arc)
            """,
            "linear_algebra": f"""
- Show vector transformations
- Use {primary_color} for vectors, {secondary_color} for transformations
- Demonstrate matrix operations visually
- Include basis vectors and coordinate systems
- Animate linear transformations
            """,
            "trigonometry": f"""
- Use unit circle approach
- Show {primary_color} for circle, {secondary_color} for trig functions
- Animate periodic function behavior
- Include angle measurements and ratios
- Connect geometric and algebraic representations
            """
        }
        
        default_spec = f"""
- Create visually appealing mathematical content
- Use {primary_color} primary, {secondary_color} secondary colors
- Include coordinate systems and mathematical notation
- Show mathematical relationships through animation
- Focus on visual mathematical beauty
        """
        
        return specs.get(domain, default_spec)
        """Get specific animation specifications for each domain."""
        specs = {
            "calculus": {
                "intro_code": """
        # Introduce calculus with a function
        plane = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": BLUE_E, "stroke_width": 1, "stroke_opacity": 0.3}
        )
        func = plane.plot(lambda x: 0.3*x**2 - 1, color=YELLOW, x_range=[-3, 3])
        
        self.play(Create(plane, rate_func=smooth), run_time=2)
        self.play(Create(func, rate_func=smooth), run_time=2)
        self.wait(0.5)
        
        self.plane = plane
        self.func = func
                """,
                "main_code": """
        # Show derivative as tangent line
        x_val = 1
        point = Dot(self.plane.c2p(x_val, 0.3*x_val**2 - 1), color=RED_B, radius=0.08)
        tangent = self.plane.plot(lambda x: 0.6*x_val*(x - x_val) + 0.3*x_val**2 - 1, 
                                 color=GREEN_B, x_range=[x_val-1, x_val+1])
        
        self.play(GrowFromCenter(point, rate_func=smooth), run_time=1)
        self.play(Create(tangent, rate_func=smooth), run_time=2)
        
        # Animate point moving along curve
        self.play(
            point.animate.move_to(self.plane.c2p(2, 0.3*4 - 1)),
            Transform(tangent, self.plane.plot(lambda x: 1.2*(x - 2) + 0.3*4 - 1, 
                                             color=GREEN_B, x_range=[1, 3])),
            rate_func=smooth,
            run_time=3
        )
        self.wait(1)
                """,
                "insight_code": """
        # Highlight the derivative concept
        derivative_eq = MathTex(r"f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}", 
                               font_size=32, color=TEAL)
        derivative_eq.to_edge(DOWN, buff=1)
        
        highlight_box = SurroundingRectangle(derivative_eq, color=YELLOW, buff=0.2)
        
        self.play(Write(derivative_eq, rate_func=smooth), run_time=2)
        self.play(Create(highlight_box, rate_func=smooth), run_time=1)
        self.wait(1)
                """,
                "requirements": """
- Must include function graphs with derivatives
- Show tangent lines as visual derivatives
- Use NumberPlane for coordinate system
- Animate limits or rate of change
- Include proper calculus notation with MathTex
                """
            },
            "algebra": {
                "intro_code": """
        # Introduce with an equation
        equation = MathTex("x^2 + 4x + 4 = 0", font_size=40, color=WHITE)
        equation.move_to(ORIGIN)
        
        self.play(Write(equation, rate_func=smooth), run_time=2)
        self.wait(1)
        
        self.equation = equation
                """,
                "main_code": """
        # Show factoring process
        factored = MathTex("(x + 2)^2 = 0", font_size=40, color=YELLOW)
        factored.move_to(self.equation.get_center())
        
        self.play(ReplacementTransform(self.equation, factored, rate_func=smooth), run_time=2)
        self.wait(1)
        
        # Show solution
        solution = MathTex("x = -2", font_size=40, color=GREEN_B)
        solution.next_to(factored, DOWN, buff=1)
        
        self.play(Write(solution, rate_func=smooth), run_time=2)
        self.wait(1)
        
        self.factored = factored
        self.solution = solution
                """,
                "insight_code": """
        # Geometric interpretation
        parabola_plane = NumberPlane(x_range=[-5, 1, 1], y_range=[-1, 8, 1])
        parabola = parabola_plane.plot(lambda x: x**2 + 4*x + 4, color=BLUE_E)
        root_point = Dot(parabola_plane.c2p(-2, 0), color=RED_B, radius=0.1)
        
        self.play(
            Create(parabola_plane, rate_func=smooth),
            Create(parabola, rate_func=smooth),
            run_time=3
        )
        self.play(GrowFromCenter(root_point, rate_func=smooth), run_time=1)
        self.wait(1)
                """,
                "requirements": """
- Must show algebraic manipulation steps
- Include geometric interpretation when possible
- Use ReplacementTransform for equation changes
- Show solutions with highlighting
- Include proper algebraic notation
                """
            },
            "geometry": {
                "intro_code": """
        # Introduce with a geometric shape
        triangle = Polygon([-2, -1, 0], [2, -1, 0], [0, 2, 0], 
                          color=BLUE_E, fill_opacity=0.3, stroke_width=3)
        
        self.play(Create(triangle, rate_func=smooth), run_time=2)
        self.wait(1)
        
        self.triangle = triangle
                """,
                "main_code": """
        # Show geometric properties
        sides = VGroup(*[
            Line(self.triangle.get_vertices()[i], 
                 self.triangle.get_vertices()[(i+1)%3], 
                 color=YELLOW, stroke_width=4)
            for i in range(3)
        ])
        
        # Add angle arcs
        angles = VGroup(*[
            Arc(radius=0.3, start_angle=0, angle=PI/3, color=TEAL)
            .move_to(self.triangle.get_vertices()[i])
            for i in range(3)
        ])
        
        self.play(Create(sides, lag_ratio=0.3, rate_func=smooth), run_time=2)
        self.play(Create(angles, lag_ratio=0.3, rate_func=smooth), run_time=2)
        self.wait(1)
        
        self.sides = sides
        self.angles = angles
                """,
                "insight_code": """
        # Show geometric theorem
        theorem = MathTex(r"\\text{Sum of angles} = 180°", 
                         font_size=36, color=GOLD)
        theorem.to_edge(DOWN, buff=1)
        
        # Highlight each angle
        for angle in self.angles:
            self.play(angle.animate.set_color(RED_B), rate_func=smooth, run_time=0.5)
            self.play(angle.animate.set_color(TEAL), rate_func=smooth, run_time=0.5)
        
        self.play(Write(theorem, rate_func=smooth), run_time=2)
        self.wait(1)
                """,
                "requirements": """
- Must include geometric constructions
- Show measurements and relationships
- Use proper geometric shapes (Circle, Polygon, Line, Arc)
- Animate step-by-step construction
- Include geometric theorems or properties
                """
            }
        }
        
        # Default specification for unknown domains
        default_spec = {
            "intro_code": """
        # General mathematical introduction
        plane = NumberPlane(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
            background_line_style={"stroke_color": BLUE_E, "stroke_width": 1, "stroke_opacity": 0.3}
        )
        
        self.play(Create(plane, rate_func=smooth), run_time=2)
        self.wait(1)
        
        self.plane = plane
            """,
            "main_code": """
        # General mathematical visualization
        spiral = ParametricFunction(
            lambda t: np.array([0.3*t*np.cos(t), 0.3*t*np.sin(t), 0]),
            t_range=[0, 4*PI], color=YELLOW
        )
        
        self.play(Create(spiral, rate_func=smooth), run_time=4)
        self.wait(1)
            """,
            "insight_code": """
        # Mathematical beauty
        beauty_text = MathTex(r"\\text{Mathematics is beautiful}", 
                             font_size=36, color=GOLD)
        beauty_text.move_to(ORIGIN)
        
        self.play(Write(beauty_text, rate_func=smooth), run_time=2)
        self.wait(1)
            """,
            "requirements": """
- Must include visual mathematical elements
- Use coordinate systems appropriately
- Show mathematical relationships
- Include proper mathematical notation
- Create visually appealing animations
            """
        }
        
        return specs.get(domain, default_spec)

    def _create_enhanced_3b1b_prompt(self):
        """Create concise prompt for 3Blue1Brown style animations."""
        system_message = SystemMessage(
            content='''
            You are a 3Blue1Brown style Manim animator. Create unique, mathematically rich animations.
            
            REQUIREMENTS:
            - Always use 5-method structure: setup_scene, introduce_concept, animate_main_content, reveal_key_insight, conclude_animation
            - Use 3B1B colors: BLUE_E, TEAL, YELLOW, RED_B, GREEN_B, GOLD
            - Always use rate_func=smooth
            - Include proper mathematical visualizations (NumberPlane, MathTex, geometric shapes)
            - Avoid text overlap - use FadeOut before creating new elements
            - Make each animation visually unique and mathematically meaningful
            
            Generate only complete Python code that compiles and runs.
            '''
        )

        human_message_prompt = HumanMessagePromptTemplate.from_template("{human_input}")

        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            human_message_prompt,
        ])
        
        return prompt
        """Create enhanced prompt for 3Blue1Brown style animations."""
        system_message = SystemMessage(
            content='''
            You are Grant Sanderson (3Blue1Brown) creating mathematical animations with Manim.
            
            Your animations are characterized by:
            
            VISUAL STORYTELLING:
            - Start with intuitive, visual introductions to abstract concepts
            - Build complexity gradually through visual layers
            - Use visual metaphors and analogies extensively
            - Create "aha moments" through well-timed reveals
            - Show multiple perspectives of the same mathematical idea
            
            ANIMATION TECHNIQUES:
            - Smooth, purposeful camera movements
            - Objects that morph and transform meaningfully
            - Synchronized animations that create rhythm
            - Strategic use of color to guide attention
            - Elegant transitions between mathematical concepts
            
            MATHEMATICAL RIGOR:
            - Every visual has mathematical meaning
            - Show formal notation alongside intuitive visuals
            - Demonstrate proofs through dynamic geometric constructions
            - Use coordinate systems and function graphs extensively
            - Make abstract concepts concrete through visualization
            
            CODE REQUIREMENTS:
            - Rich use of NumberPlane, Axes, FunctionGraph
            - Dynamic transformations with Transform, ReplacementTransform
            - Geometric constructions with Circle, Polygon, Line, Arc
            - Mathematical typography with MathTex and proper LaTeX
            - Smooth animations with appropriate rate_func
            - Strategic use of 3B1B color palette (BLUE_E, TEAL, YELLOW, etc.)
            - Camera movements and zooming for emphasis
            - Multiple simultaneous animations for richness
            
            Create animations that make viewers say "I never thought of it that way!"
            Every animation should reveal mathematical beauty and insight.
            
            Generate only executable Python code with rich mathematical visualizations.
            Make mathematics come alive through motion, color, and geometric beauty.
            '''
        )

        human_message_prompt = HumanMessagePromptTemplate.from_template("{human_input}")

        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            human_message_prompt,
        ])
        
        return prompt

    def _clean_and_enhance_code(self, raw_code):
        """Enhanced code cleaning for consistent 3Blue1Brown style animations."""
        print(f"Raw LLM response length: {len(raw_code)}")
        
        # Remove markdown blocks and extract code
        code = re.sub(r'^```[a-zA-Z]*\s*', '', raw_code.strip(), flags=re.MULTILINE)
        code = re.sub(r'\s*```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'```', '', code).strip()
        
        if not code or len(code) < 200:  # Increased minimum length for structured code
            print("Code too short or empty")
            return ""
        
        # Ensure proper imports
        if 'from manim import *' not in code:
            code = 'from manim import *\n' + code
        if 'import numpy as np' not in code:
            code = code.replace('from manim import *', 'from manim import *\nimport numpy as np')
        
        # Fix common syntax issues
        code = self._fix_string_literals(code)
        code = self._validate_latex_syntax(code)
        code = self._fix_method_structure(code)
        code = self._enhance_3b1b_syntax(code)
        
        # Validate structure
        if not self._validate_structured_code(code):
            print("Code failed structure validation")
            return ""
        
        # Validate LaTeX syntax
        code = self._validate_latex_syntax(code)
        
        return code

    def _fix_string_literals(self, code):
        """Fix common string literal issues that cause syntax errors while preserving LaTeX."""
        lines = code.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Check for unescaped multi-line strings in Text() calls (not MathTex - preserve LaTeX)
            if 'Text(' in line and '"' in line and 'MathTex(' not in line:
                # Find the start of the string
                start_quote = line.find('"')
                if start_quote != -1:
                    # Check if this looks like an unterminated string
                    end_quote = line.find('"', start_quote + 1)
                    if end_quote == -1:
                        # Likely an unterminated string, try to fix it
                        # Look for the next line that ends the string
                        j = i + 1
                        text_content = line[start_quote + 1:]
                        while j < len(lines) and '", ' not in lines[j] and lines[j].strip() != '",':
                            text_content += ' ' + lines[j].strip()
                            j += 1
                        
                        if j < len(lines):
                            # Found the end, reconstruct the line
                            # Clean the text content (but preserve basic formatting)
                            text_content = text_content.replace('\n', ' ').replace('\r', ' ')
                            text_content = re.sub(r'\s+', ' ', text_content).strip()
                            if text_content.endswith('"'):
                                text_content = text_content[:-1]
                            
                            # Only escape quotes, don't touch backslashes (for LaTeX compatibility)
                            text_content = text_content.replace('"', '\\"')
                            
                            # Limit length to prevent overly long strings
                            if len(text_content) > 100:
                                text_content = text_content[:97] + "..."
                            
                            # Reconstruct the line
                            before_quote = line[:start_quote]
                            after_content = lines[j][lines[j].find('",'):]
                            new_line = f'{before_quote}"{text_content}"{after_content}'
                            fixed_lines.append(new_line)
                            
                            # Skip the processed lines
                            i = j
                            continue
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def _fix_method_structure(self, code):
        """Ensure the required method structure is present."""
        required_methods = [
            'def setup_scene(self)',
            'def introduce_concept(self)',
            'def animate_main_content(self)',
            'def reveal_key_insight(self)',
            'def conclude_animation(self)'
        ]
        
        # Check if all required methods are present
        missing_methods = []
        for method in required_methods:
            if method not in code:
                missing_methods.append(method)
        
        # Add missing methods with basic implementations
        for method in missing_methods:
            method_name = method.replace('def ', '').replace('(self)', '')
            basic_implementation = f"""
    {method}:
        \"\"\"Required method - basic implementation\"\"\"
        # Basic animation for {method_name}
        placeholder = Text("Mathematical Concept", font_size=36, color=BLUE_E)
        self.play(Write(placeholder, rate_func=smooth), run_time=2)
        self.wait(1)
        self.play(FadeOut(placeholder), run_time=1)
"""
            # Insert before the last method or at the end of the class
            if 'def conclude_animation(self):' in code:
                code = code.replace('def conclude_animation(self):', basic_implementation + '\n    def conclude_animation(self):')
            else:
                # Add at the end of the class
                code = code.rstrip() + basic_implementation
        
        return code

    def _enhance_3b1b_syntax(self, code):
        """Enhance code with 3Blue1Brown specific improvements."""
        lines = code.split('\n')
        enhanced_lines = []
        
        for line in lines:
            # Replace basic colors with 3B1B palette
            line = line.replace('BLUE', 'BLUE_E')
            line = line.replace('RED', 'RED_B')
            line = line.replace('GREEN', 'GREEN_B')
            
            # Enhance animation calls
            if 'self.play(' in line and 'run_time' not in line and not line.strip().endswith(','):
                line = line.rstrip(')') + ', run_time=1.5)'
            
            # Add rate functions for smoother animations
            if 'Transform(' in line and 'rate_func' not in line:
                line = line.rstrip(')') + ', rate_func=smooth)'
            
            enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)

    def _validate_structured_code(self, code):
        """Validate that the code follows the required structure."""
        # Check for class definition
        if not re.search(r'class\s+\w+Scene\(Scene\):', code):
            print("Missing proper class definition")
            return False
        
        # Check for required methods
        required_methods = [
            'def construct(self)',
            'def setup_scene(self)',
            'def introduce_concept(self)',
            'def animate_main_content(self)',
            'def reveal_key_insight(self)',
            'def conclude_animation(self)'
        ]
        
        for method in required_methods:
            if method not in code:
                print(f"Missing required method: {method}")
                return False
        
        # Check for mathematical visual content
        visual_indicators = [
            'MathTex', 'NumberPlane', 'Axes', 'FunctionGraph', 'Circle', 
            'Transform', 'Create', 'self.play', 'Vector', 'Line', 'Polygon'
        ]
        
        visual_count = sum(1 for indicator in visual_indicators if indicator in code)
        if visual_count < 3:
            print(f"Insufficient visual mathematical content (found {visual_count}, need 3+)")
            return False
        
        # Check for 3B1B color usage
        color_indicators = ['BLUE_E', 'TEAL', 'YELLOW', 'RED_B', 'GREEN_B', 'GOLD']
        color_count = sum(1 for color in color_indicators if color in code)
        if color_count < 2:
            print(f"Insufficient 3B1B color usage (found {color_count}, need 2+)")
            return False
        
        # Basic syntax validation
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError as e:
            print(f"Syntax error in generated code: {e}")
            return False
        except Exception as e:
            print(f"Other error in code validation: {e}")
            return False

    def _validate_latex_syntax(self, code):
        """Validate and fix common LaTeX syntax issues in MathTex calls."""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            if 'MathTex(' in line:
                # Common LaTeX fixes
                # Ensure proper escaping for backslashes in raw strings
                if not line.strip().startswith('r"') and not line.strip().startswith("r'"):
                    # Check if this needs raw string conversion
                    if '\\' in line and not line.count('\\\\') == line.count('\\'):
                        # Convert to raw string if not already
                        line = line.replace('MathTex("', 'MathTex(r"')
                        line = line.replace("MathTex('", "MathTex(r'")
                
                # Fix common LaTeX command issues
                # Ensure \frac, \text, etc. are properly preserved
                line = line.replace('rac{', '\\frac{')  # Fix corrupted \frac
                line = line.replace('ext{', '\\text{')  # Fix corrupted \text
                line = line.replace('im{', '\\lim{')    # Fix corrupted \lim
                line = line.replace('rac', '\\frac')    # Fix standalone corrupted frac
                
                # Ensure proper spacing in LaTeX
                line = re.sub(r'\\(\w+)([a-zA-Z])', r'\\\1 \2', line)
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def _generate_enhanced_3b1b_fallback(self, structured_content):
        """Generate varied, structured 3Blue1Brown style fallback code."""
        title = structured_content.get('title', 'Mathematical Concept')
        domain = structured_content.get('mathematical_domain', 'general')
        variation = structured_content.get('_variation', 0)
        
        # Clean the title to prevent syntax errors
        title = title.replace('\n', ' ').replace('\r', ' ').strip()
        title = re.sub(r'\s+', ' ', title)
        if len(title) > 35:
            title = title[:32] + "..."
        
        class_name = re.sub(r'[^a-zA-Z0-9]', '', title.replace(' ', ''))
        if not class_name:
            class_name = "MathAnimation"
        
        # Create variations based on domain and variation number
        templates = self._get_varied_templates(domain, variation)
        template = templates[variation % len(templates)]
        
        # Escape the title properly for Python string
        escaped_title = title.replace('"', '\\"').replace("'", "\\'")
        
        return template.format(
            class_name=class_name,
            escaped_title=escaped_title,
            variation=variation
        )

    def _get_varied_templates(self, domain, variation):
        """Get varied animation templates to prevent repetition."""
        
        if domain == "calculus":
            return [
                # Template 1: Function and derivative
                """from manim import *
import numpy as np

class {class_name}Scene(Scene):
    def construct(self):
        self.setup_scene()
        self.introduce_concept()
        self.animate_main_content()
        self.reveal_key_insight()
        self.conclude_animation()
    
    def setup_scene(self):
        self.camera.background_color = "#0d1117"
        title = Text("{escaped_title}", font_size=40, color=BLUE_E, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title, rate_func=smooth), run_time=1.5)
        self.play(title.animate.scale(0.6).to_corner(UL), run_time=1)
        self.title = title
    
    def introduce_concept(self):
        axes = Axes(x_range=[-3, 3, 1], y_range=[-2, 4, 1], 
                   axis_config={{"color": TEAL}})
        func = axes.plot(lambda x: x**2, color=YELLOW)
        
        self.play(Create(axes, rate_func=smooth), run_time=2)
        self.play(Create(func, rate_func=smooth), run_time=2)
        self.axes, self.func = axes, func
    
    def animate_main_content(self):
        # Show derivative at specific point
        x_val = 1.5
        point = Dot(self.axes.c2p(x_val, x_val**2), color=RED_B, radius=0.08)
        tangent = self.axes.plot(lambda x: 2*x_val*(x-x_val) + x_val**2, 
                               color=GREEN_B, x_range=[x_val-1, x_val+1])
        
        self.play(GrowFromCenter(point, rate_func=smooth), run_time=1)
        self.play(Create(tangent, rate_func=smooth), run_time=2)
        
        # Move point and update tangent
        new_x = -1
        self.play(
            point.animate.move_to(self.axes.c2p(new_x, new_x**2)),
            Transform(tangent, self.axes.plot(lambda x: 2*new_x*(x-new_x) + new_x**2, 
                                            color=GREEN_B, x_range=[new_x-1, new_x+1])),
            rate_func=smooth, run_time=3
        )
        self.point, self.tangent = point, tangent
    
    def reveal_key_insight(self):
        # Clear previous elements properly
        self.play(FadeOut(self.axes), FadeOut(self.func), 
                 FadeOut(self.point), FadeOut(self.tangent), run_time=1)
        
        derivative_eq = MathTex(r"\\frac{{df}}{{dx}} = \\lim_{{h \\to 0}} \\frac{{f(x+h) - f(x)}}{{h}}", 
                               font_size=28, color=TEAL)
        derivative_eq.move_to(ORIGIN)
        
        self.play(Write(derivative_eq, rate_func=smooth), run_time=2)
        self.derivative_eq = derivative_eq
    
    def conclude_animation(self):
        summary = MathTex(r"\\text{{Derivative = Instantaneous Rate}}", 
                         font_size=32, color=GOLD)
        summary.next_to(self.derivative_eq, DOWN, buff=1)
        
        self.play(Write(summary, rate_func=smooth), run_time=2)
        self.wait(1)
        
        self.play(*[FadeOut(mob, shift=UP*0.3) for mob in self.mobjects], 
                 run_time=2, rate_func=smooth)
""",
                # Template 2: Integral visualization
                """from manim import *
import numpy as np

class {class_name}Scene(Scene):
    def construct(self):
        self.setup_scene()
        self.introduce_concept()
        self.animate_main_content()
        self.reveal_key_insight()
        self.conclude_animation()
    
    def setup_scene(self):
        self.camera.background_color = "#0d1117"
        title = Text("{escaped_title}", font_size=40, color=GREEN_B, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title, rate_func=smooth), run_time=1.5)
        self.play(title.animate.scale(0.6).to_corner(UR), run_time=1)
        self.title = title
    
    def introduce_concept(self):
        plane = NumberPlane(x_range=[-2, 4, 1], y_range=[-1, 3, 1],
                           background_line_style={{"stroke_color": BLUE_E, "stroke_opacity": 0.3}})
        curve = plane.plot(lambda x: 0.5*x + 1, color=YELLOW, x_range=[0, 3])
        
        self.play(Create(plane, rate_func=smooth), run_time=2)
        self.play(Create(curve, rate_func=smooth), run_time=2)
        self.plane, self.curve = plane, curve
    
    def animate_main_content(self):
        # Show area under curve
        area = plane.get_area(self.curve, x_range=[0, 3], color=TEAL, opacity=0.5)
        
        self.play(Create(area, rate_func=smooth), run_time=3)
        
        # Add Riemann rectangles
        rectangles = plane.get_riemann_rectangles(
            self.curve, x_range=[0, 3], dx=0.5, color=RED_B, opacity=0.7
        )
        
        self.play(Create(rectangles, lag_ratio=0.1, rate_func=smooth), run_time=2)
        self.area, self.rectangles = area, rectangles
    
    def reveal_key_insight(self):
        # Clear and show integral
        self.play(FadeOut(self.rectangles), run_time=1)
        
        integral_eq = MathTex(r"\\int_{{a}}^{{b}} f(x) \\, dx = \\text{{Area}}", 
                             font_size=32, color=GOLD)
        integral_eq.to_edge(DOWN, buff=1)
        
        self.play(Write(integral_eq, rate_func=smooth), run_time=2)
        self.integral_eq = integral_eq
    
    def conclude_animation(self):
        summary = Text("Integration = Accumulated Change", font_size=28, color=RED_B)
        summary.next_to(self.integral_eq, UP, buff=0.5)
        
        self.play(Write(summary, rate_func=smooth), run_time=2)
        self.wait(1)
        
        self.play(*[FadeOut(mob, shift=DOWN*0.3) for mob in self.mobjects], 
                 run_time=2, rate_func=smooth)
"""
            ]
        
        elif domain == "algebra":
            return [
                # Template 1: Equation solving
                """from manim import *
import numpy as np

class {class_name}Scene(Scene):
    def construct(self):
        self.setup_scene()
        self.introduce_concept()
        self.animate_main_content()
        self.reveal_key_insight()
        self.conclude_animation()
    
    def setup_scene(self):
        self.camera.background_color = "#0d1117"
        title = Text("{escaped_title}", font_size=40, color=YELLOW, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title, rate_func=smooth), run_time=1.5)
        self.play(title.animate.scale(0.6).to_corner(UL), run_time=1)
        self.title = title
    
    def introduce_concept(self):
        equation = MathTex("2x + 6 = 14", font_size=44, color=WHITE)
        equation.move_to(ORIGIN)
        
        self.play(Write(equation, rate_func=smooth), run_time=2)
        self.equation = equation
    
    def animate_main_content(self):
        # Step 1: Subtract 6
        step1 = MathTex("2x + 6 - 6 = 14 - 6", font_size=40, color=TEAL)
        step1.move_to(self.equation.get_center())
        
        self.play(ReplacementTransform(self.equation, step1, rate_func=smooth), run_time=2)
        self.wait(1)
        
        # Step 2: Simplify
        step2 = MathTex("2x = 8", font_size=44, color=GREEN_B)
        step2.move_to(step1.get_center())
        
        self.play(ReplacementTransform(step1, step2, rate_func=smooth), run_time=2)
        self.wait(1)
        
        # Step 3: Divide by 2
        step3 = MathTex("x = 4", font_size=48, color=GOLD)
        step3.move_to(step2.get_center())
        
        self.play(ReplacementTransform(step2, step3, rate_func=smooth), run_time=2)
        self.step3 = step3
    
    def reveal_key_insight(self):
        # Show verification
        verification = MathTex("\\text{{Check: }} 2(4) + 6 = 14 \\, \\checkmark", 
                              font_size=32, color=RED_B)
        verification.next_to(self.step3, DOWN, buff=1)
        
        self.play(Write(verification, rate_func=smooth), run_time=2)
        self.verification = verification
    
    def conclude_animation(self):
        conclusion = Text("Algebra: Balance and Solve", font_size=32, color=BLUE_E)
        conclusion.to_edge(DOWN, buff=1)
        
        self.play(Write(conclusion, rate_func=smooth), run_time=2)
        self.wait(1)
        
        self.play(*[FadeOut(mob, shift=LEFT*0.5) for mob in self.mobjects], 
                 run_time=2, rate_func=smooth)
"""
            ]
        
        elif domain == "geometry":
            return [
                # Template 1: Triangle construction
                """from manim import *
import numpy as np

class {class_name}Scene(Scene):
    def construct(self):
        self.setup_scene()
        self.introduce_concept()
        self.animate_main_content()
        self.reveal_key_insight()
        self.conclude_animation()
    
    def setup_scene(self):
        self.camera.background_color = "#0d1117"
        title = Text("{escaped_title}", font_size=40, color=TEAL, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title, rate_func=smooth), run_time=1.5)
        self.play(title.animate.scale(0.6).to_corner(UL), run_time=1)
        self.title = title
    
    def introduce_concept(self):
        # Create triangle vertices
        A = np.array([-2, -1, 0])
        B = np.array([2, -1, 0])
        C = np.array([0, 2, 0])
        
        triangle = Polygon(A, B, C, color=BLUE_E, fill_opacity=0.3, stroke_width=3)
        
        self.play(Create(triangle, rate_func=smooth), run_time=2)
        self.triangle = triangle
        self.vertices = [A, B, C]
    
    def animate_main_content(self):
        # Add vertex labels
        labels = VGroup(*[
            MathTex(label, font_size=32, color=YELLOW).next_to(vertex, direction, buff=0.2)
            for label, vertex, direction in zip(["A", "B", "C"], self.vertices, [DL, DR, UP])
        ])
        
        self.play(Write(labels, lag_ratio=0.3, rate_func=smooth), run_time=2)
        
        # Show sides
        sides = VGroup(*[
            Line(self.vertices[i], self.vertices[(i+1)%3], color=GREEN_B, stroke_width=4)
            for i in range(3)
        ])
        
        self.play(Create(sides, lag_ratio=0.2, rate_func=smooth), run_time=2)
        
        # Add side lengths
        side_labels = VGroup(
            MathTex "a", font_size=28, color=RED_B).next_to(sides[0], DOWN),
            MathTex "b", font_size=28, color=RED_B).next_to(sides[1], RIGHT),
            MathTex "c", font_size=28, color=RED_B).next_to(sides[2], LEFT)
        )
        
        self.play(Write(side_labels, lag_ratio=0.2, rate_func=smooth), run_time=1.5)
        self.labels, self.sides, self.side_labels = labels, sides, side_labels
    
    def reveal_key_insight(self):
        # Show triangle inequality
        inequality = MathTex("a + b > c", font_size=36, color=GOLD)
        inequality.to_edge(DOWN, buff=1)
        
        highlight_box = SurroundingRectangle(inequality, color=YELLOW, buff=0.2)
        
        self.play(Write(inequality, rate_func=smooth), run_time=2)
        self.play(Create(highlight_box, rate_func=smooth), run_time=1)
        self.inequality, self.highlight_box = inequality, highlight_box
    
    def conclude_animation(self):
        summary = Text("Geometry: Shape and Space", font_size=28, color=TEAL)
        summary.next_to(self.inequality, UP, buff=0.5)
        
        self.play(Write(summary, rate_func=smooth), run_time=2)
        self.wait(1)
        
        self.play(*[FadeOut(mob, shift=RIGHT*0.5) for mob in self.mobjects], 
                 run_time=2, rate_func=smooth)
"""
            ]
        
        # Default/general template
        return [
            """from manim import *
import numpy as np

class {class_name}Scene(Scene):
    def construct(self):
        self.setup_scene()
        self.introduce_concept()
        self.animate_main_content()
        self.reveal_key_insight()
        self.conclude_animation()
    
    def setup_scene(self):
        self.camera.background_color = "#0d1117"
        title = Text("{escaped_title}", font_size=40, color=BLUE_E, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title, rate_func=smooth), run_time=1.5)
        self.play(title.animate.scale(0.6).to_corner(UL), run_time=1)
        self.title = title
    
    def introduce_concept(self):
        # Mathematical spiral
        spiral = ParametricFunction(
            lambda t: np.array([0.2*t*np.cos(t), 0.2*t*np.sin(t), 0]),
            t_range=[0, 6*PI], color=YELLOW
        )
        
        self.play(Create(spiral, rate_func=smooth), run_time=3)
        self.spiral = spiral
    
    def animate_main_content(self):
        # Golden ratio visualization
        phi_eq = MathTex("\\phi = \\frac{{1 + \\sqrt{{5}}}}{{2}}", font_size=36, color=GOLD)
        phi_eq.to_edge(DOWN, buff=2)
        
        self.play(Write(phi_eq, rate_func=smooth), run_time=2)
        
        # Add circles at spiral points
        circles = VGroup(*[
            Circle(radius=0.05, color=TEAL, fill_opacity=0.8).move_to(
                np.array([0.2*t*np.cos(t), 0.2*t*np.sin(t), 0])
            ) for t in np.linspace(0, 6*PI, 20)
        ])
        
        self.play(Create(circles, lag_ratio=0.05, rate_func=smooth), run_time=2)
        self.phi_eq, self.circles = phi_eq, circles
    
    def reveal_key_insight(self):
        insight = Text("Mathematics: Pattern and Beauty", font_size=32, color=RED_B)
        insight.move_to(ORIGIN)
        
        self.play(Write(insight, rate_func=smooth), run_time=2)
        self.insight = insight
    
    def conclude_animation(self):
        final_text = MathTex("\\text{{Variation {variation}: Mathematical Wonder}}", 
                           font_size=28, color=GREEN_B)
        final_text.next_to(self.insight, DOWN, buff=1)
        
        self.play(Write(final_text, rate_func=smooth), run_time=2)
        self.wait(1)
        
        self.play(*[FadeOut(mob, shift=UP*0.3) for mob in self.mobjects], 
                 run_time=2, rate_func=smooth)
"""
        ]

# Initialize the enhanced 3Blue1Brown style generator
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
manim_code_generator = Enhanced3B1BManimGenerator(GROQ_API_KEY)