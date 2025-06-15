import os
from dotenv import load_dotenv
import logging
import re
import json
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logging.getLogger('comtypes').setLevel(logging.WARNING)

load_dotenv('.env')

class ManimGenerator:
    """Enhanced Manim code generator that creates error-free, step-by-step educational animations"""
    
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
        self.groq_chat = ChatGroq(groq_api_key=self.groq_api_key, model_name='qwen-qwq-32b')
        
        # Layout management for preventing overlaps
        self.layout_zones = self._initialize_layout_zones()
        self.animation_templates = self._initialize_animation_templates()
        
        self.prompt = self._create_advanced_prompt()
        self.conversation = ConversationChain(
            llm=self.groq_chat,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory,
            input_key="human_input",
        )

    def _initialize_layout_zones(self):
        """Define non-overlapping screen zones for different content types"""
        return {
            'title_zone': {'position': 'UP*3.2', 'buffer': 0.3, 'font_size': 52},
            'subtitle_zone': {'position': 'UP*2.5', 'buffer': 0.2, 'font_size': 36},
            'step_title_zone': {'position': 'UP*2.8', 'buffer': 0.3, 'font_size': 40},
            'main_equation_zone': {'position': 'UP*1.2', 'buffer': 0.4, 'font_size': 36},
            'secondary_equation_zone': {'position': 'ORIGIN', 'buffer': 0.3, 'font_size': 28},
            'visual_center_zone': {'position': 'ORIGIN', 'buffer': 0.5, 'scale': 1.0},
            'left_panel_zone': {'position': 'LEFT*4.5', 'buffer': 0.3, 'font_size': 24},
            'right_panel_zone': {'position': 'RIGHT*4.5', 'buffer': 0.3, 'font_size': 24},
            'bottom_info_zone': {'position': 'DOWN*2.8', 'buffer': 0.2, 'font_size': 24},
            'insight_zone': {'position': 'UP*0.5', 'buffer': 0.5, 'font_size': 32},
            'axes_zone': {'x_range': [-6, 6, 1], 'y_range': [-3.5, 3.5, 1]}
        }

    def _initialize_animation_templates(self):
        """Pre-defined animation sequences for different content types"""
        return {
            'title_entrance': {
                'animation': 'Write',
                'rate_func': 'smooth',
                'run_time': 2.0,
                'follow_up': 'self.wait(0.8)'
            },
            'equation_reveal': {
                'animation': 'Write',
                'rate_func': 'smooth',
                'run_time': 2.5,
                'follow_up': 'self.wait(1.0)'
            },
            'graph_creation': {
                'animation': 'Create',
                'rate_func': 'smooth',
                'run_time': 3.0,
                'follow_up': 'self.wait(1.2)'
            },
            'shape_draw': {
                'animation': 'DrawBorderThenFill',
                'rate_func': 'smooth',
                'run_time': 2.0,
                'follow_up': 'self.wait(0.8)'
            },
            'transformation': {
                'animation': 'Transform',
                'rate_func': 'smooth',
                'run_time': 2.5,
                'follow_up': 'self.wait(1.0)'
            },
            'fade_transition': {
                'animation': 'FadeOut',
                'rate_func': 'smooth',
                'run_time': 1.5,
                'follow_up': 'self.wait(0.5)'
            }
        }

    def generate_3b1b_manim_code(self, structured_content):
        """
        Generate enhanced 3Blue1Brown style Manim code from structured educational content
        
        Args:
            structured_content (dict): Rich educational content structure from script generator
        """
        if not structured_content:
            return self._get_minimal_fallback()

        try:
            # Extract and validate content structure
            educational_breakdown = structured_content.get('educational_breakdown', {})
            manim_structure = structured_content.get('manim_structure', {})
            
            if not educational_breakdown:
                logging.warning("No educational breakdown found, using fallback")
                return self._get_minimal_fallback()
            
            # Extract key information
            title = educational_breakdown.get('title', 'Mathematical Concept')
            educational_steps = educational_breakdown.get('educational_steps', [])
            topic_analysis = educational_breakdown.get('topic_analysis', {})
            domain = topic_analysis.get('domain', 'Mathematics')
            
            # Create comprehensive prompt with step-by-step structure
            manim_prompt = self._create_comprehensive_prompt(
                title, domain, educational_steps, manim_structure
            )
            
            # Generate code
            response = self.conversation.predict(human_input=manim_prompt)
            
            # Enhanced validation and cleanup
            validated_code = self._advanced_code_validation(response, educational_steps)
            
            return validated_code

        except Exception as e:
            logging.error(f"Error generating enhanced Manim code: {e}")
            return self._get_comprehensive_fallback(educational_steps)

    def _create_comprehensive_prompt(self, title, domain, educational_steps, manim_structure):
        """Create a content-aware prompt that actually uses the educational breakdown"""
        
        clean_title = ''.join(c for c in title if c.isalnum())[:20] or "MathConcept"
        
        # Extract ACTUAL content from educational steps
        content_analysis = self._extract_actual_content(educational_steps)
        
        # Build SPECIFIC animation instructions based on actual content
        specific_instructions = self._build_content_specific_instructions(educational_steps, content_analysis)
        
        # Create zone-based positioning instructions
        zone_instructions = self._generate_zone_instructions()
        
        # Generate step method calls safely
        step_method_calls = '\n        '.join([f'self.animate_step_{i+1}()' for i in range(len(educational_steps))])
        
        return """Create a 3Blue1Brown-style Manim animation for: "{title}"

TOPIC DOMAIN: {domain}
TOTAL EDUCATIONAL STEPS: {step_count}

ACTUAL CONTENT TO VISUALIZE:
{specific_instructions}

CONTENT-SPECIFIC REQUIREMENTS:

1. IMPLEMENTATION MANDATE - Use ACTUAL Content:
   - EVERY equation from the educational breakdown MUST be displayed
   - EVERY key concept MUST be visualized appropriately
   - EVERY animation plan MUST be implemented as specified
   - NO generic animations - follow the specific content requirements above

2. EXACT CLASS STRUCTURE:
```python
from manim import *
import numpy as np

class {clean_title}Scene(Scene):
    def construct(self):
        self.setup_educational_environment()
        self.animate_step_by_step()
        self.create_final_summary()
        
    def setup_educational_environment(self):
        # Set 3B1B signature style
        self.camera.background_color = "#0f0f23"
        
        # Content-appropriate coordinate system
        self.axes = Axes(
            x_range=[-6, 6, 1], 
            y_range=[-3.5, 3.5, 1],
            axis_config={{"color": BLUE_E, "stroke_width": 2}}
        )
        self.axes.set_opacity(0.3)
        self.add(self.axes)
        
        # Initialize content tracking
        self.current_objects = {{}}
        self.step_number = 0
        
    def animate_step_by_step(self):
        # MUST implement each step exactly as specified in content requirements
        {step_method_calls}
        
    def clear_step(self, keep_axes=True):
        objects_to_clear = [obj for obj in self.mobjects if obj != self.axes]
        if objects_to_clear:
            self.play(*[FadeOut(obj) for obj in objects_to_clear], run_time=1.2)
            self.remove(*objects_to_clear)
        self.current_objects.clear()
        self.wait(0.5)
```

3. ZONE-BASED POSITIONING (PREVENTS OVERLAPS):
{zone_instructions}

4. CONTENT-DRIVEN ANIMATION REQUIREMENTS:
- MANDATORY: Each animate_step_X() method MUST implement the specific content requirements listed above
- MANDATORY: Display all equations exactly as provided in the educational breakdown
- MANDATORY: Visualize all key concepts with appropriate mathematical/scientific representations
- MANDATORY: Follow the specific animation plans provided for each step
- NO GENERIC CONTENT: Every element must match the actual educational content

5. STEP-BY-STEP IMPLEMENTATION GUIDE:
- Create separate method for each step: animate_step_1(), animate_step_2(), etc.
- Each method must start with self.clear_step() to prevent overlaps
- Use the specific step title, description, and concepts provided
- Implement the exact visual elements specified for each step
- Follow the narration context to create appropriate animations

6. MATHEMATICAL/SCIENTIFIC VISUALIZATION RULES:
- Functions: Use self.axes.plot() for any mathematical functions mentioned
- Equations: Create MathTex objects for all equations from educational content
- Geometric shapes: Create specific shapes mentioned in key concepts
- Scientific phenomena: Use appropriate Manim objects (vectors, waves, particles)
- Real-world examples: Visualize using relevant analogies and representations

7. CONTENT-SPECIFIC ANIMATION TIMING:
- Title reveal: run_time=2.0, rate_func=smooth
- Equation display: run_time=2.5, rate_func=smooth  
- Concept visualization: run_time=3.0, rate_func=smooth
- Transformations: run_time=2.5, rate_func=smooth
- Always add self.wait(1.0) after important revelations

8. 3BLUE1BROWN COLOR SCHEME:
- BLUE_E: Primary titles and coordinate systems
- TEAL_E: Step titles and mathematical labels  
- YELLOW_E: Key equations and mathematical highlights
- RED_E: Important emphasis and warnings
- GREEN_E: Positive results and conclusions
- GOLD: Key insights and "aha" moments
- WHITE: General text and descriptions
- ORANGE: Real-world connections and examples

9. ERROR PREVENTION AND VALIDATION:
- Every variable must be defined before use
- All mathematical expressions must be domain-safe
- Font sizes must be specified for all Text objects
- Positions must use the zone system to prevent overlaps
- All objects must be stored as instance variables

10. CONTENT ENGAGEMENT REQUIREMENTS:
- Highlight key concepts using Indicate() and ShowPassingFlash()
- Create dramatic reveals for important equations
- Use progressive build-up following the educational sequence
- Add visual emphasis for insights and connections
- Include smooth transitions that maintain educational flow

CRITICAL OUTPUT REQUIREMENTS:
1. GENERATE COMPLETE, EXECUTABLE PYTHON CODE ONLY
2. NO MARKDOWN BLOCKS, NO EXPLANATIONS, NO COMMENTS OUTSIDE CODE
3. EVERY LINE MUST BE SYNTACTICALLY CORRECT
4. IMPLEMENT THE SPECIFIC EDUCATIONAL CONTENT - NOT GENERIC TEMPLATES

CONTENT FIDELITY MANDATE:
- Use the EXACT equations, concepts, and visual elements specified above
- Create animations that directly correspond to the educational breakdown
- Ensure each step implements its specific content requirements
- NO placeholder content - everything must be derived from the actual educational material

Generate the complete Manim Scene class that brings this specific educational content to life through precise, engaging animations.""".format(
            title=title,
            domain=domain, 
            step_count=len(educational_steps),
            specific_instructions=specific_instructions,
            clean_title=clean_title,
            step_method_calls=step_method_calls,
            zone_instructions=zone_instructions
        )

    def _analyze_educational_steps(self, educational_steps):
        """Analyze educational steps to determine optimal animation strategies"""
        analysis = {
            'total_steps': len(educational_steps),
            'complexity_levels': [],
            'visual_types': [],
            'mathematical_content': [],
            'animation_strategies': []
        }
        
        for step in educational_steps:
            # Analyze complexity
            if 'difficulty_level' in step:
                analysis['complexity_levels'].append(step['difficulty_level'])
            
            # Determine visual approach based on content
            key_concepts = step.get('key_concepts', [])
            equations = step.get('equations', [])
            
            if equations:
                analysis['mathematical_content'].append('equations')
                analysis['visual_types'].append('equation_focused')
            
            if any(concept in ['graph', 'function', 'plot'] for concept in key_concepts):
                analysis['visual_types'].append('graph_based')
            
            if any(concept in ['geometric', 'shape', 'circle', 'triangle'] for concept in key_concepts):
                analysis['visual_types'].append('geometric')
                
            # Determine animation strategy
            if step.get('step_number', 0) <= 2:
                analysis['animation_strategies'].append('gentle_introduction')
            elif 'insight' in step.get('description', '').lower():
                analysis['animation_strategies'].append('dramatic_reveal')
            else:
                analysis['animation_strategies'].append('progressive_build')
        
        return analysis

    def _generate_zone_instructions(self):
        """Generate specific positioning instructions for each zone"""
        zones = self.layout_zones
        instructions = []
        
        for zone_name, zone_config in zones.items():
            if 'position' in zone_config:
                instructions.append("   - {}: {}, font_size={}".format(
                    zone_name, zone_config['position'], zone_config.get('font_size', 28)
                ))
        
        return "\n".join(instructions)

    def _build_step_instructions(self, educational_steps, step_analysis):
        """Build detailed instructions for each educational step"""
        instructions = []
        
        for i, step in enumerate(educational_steps, 1):
            step_title = step.get('step_title', f'Step {i}')
            description = step.get('description', '')
            key_concepts = step.get('key_concepts', [])
            equations = step.get('equations', [])
            
            instruction = """
STEP {}: {}
- Description: {}...
- Key Concepts: {}
- Equations: {}
- Animation Method: animate_step_{}()
- Visual Focus: {}
""".format(
                i, step_title, description[:100], 
                ', '.join(key_concepts[:3]), ', '.join(equations[:2]), i,
                'equations' if equations else 'concepts' if key_concepts else 'general'
            )
            instructions.append(instruction)
        
        return "\n".join(instructions)

    def _create_advanced_prompt(self):
        """Create the advanced system prompt for content-driven code generation"""
        system_message = SystemMessage(
            content="""
You are an expert Manim developer specializing in content-driven 3Blue1Brown-style educational animations.

CORE PRINCIPLES:
1. Generate CONTENT-SPECIFIC animations based on actual educational material provided
2. NO generic templates or placeholder content - use the exact concepts, equations, and requirements specified
3. Create mathematically and scientifically accurate visualizations
4. Ensure every animation directly corresponds to the educational breakdown
5. Follow 3Blue1Brown aesthetic while being content-authentic

CONTENT FIDELITY REQUIREMENTS:
- Read and implement the SPECIFIC educational content provided in each prompt
- Use the EXACT equations, formulas, and mathematical expressions given
- Visualize the SPECIFIC key concepts mentioned in each step
- Follow the DETAILED animation plans and visual requirements
- Create representations that match the scientific/mathematical domain

MANDATORY IMPLEMENTATION APPROACH:
- Analyze each educational step for its unique content requirements
- Extract specific mathematical expressions, scientific phenomena, or concepts
- Create targeted visualizations for each piece of content
- Avoid generic animations - every element must serve the educational goal
- Ensure mathematical accuracy and scientific validity

VISUAL CONTENT MAPPING:
- Mathematical functions → Interactive plots with accurate representations
- Geometric concepts → Precise shapes with correct measurements and properties
- Physics phenomena → Accurate simulations with proper physical behavior
- Chemical processes → Molecular representations with correct bonding
- Equations → Step-by-step derivations and transformations
- Real-world examples → Relevant analogies and practical applications

ERROR PREVENTION:
- Validate all mathematical expressions against their intended domains
- Ensure scientific accuracy in all representations
- Use correct units, scales, and proportions
- Verify that visualizations match the educational content
- Test all mathematical functions for domain safety
- Verify that visualizations match the educational content
- Test all mathematical functions for domain safety
- Include proper error handling in mathematical functions

ANIMATION AUTHENTICITY:
- Every animation must serve a specific educational purpose
- Use content-appropriate timing and pacing
- Create mathematically accurate transformations
- Ensure visual metaphors align with the scientific concepts
- Build animations that enhance understanding of the specific content

OUTPUT MANDATE: Generate complete, syntactically correct Python code that creates content-authentic, educationally accurate animations that directly implement the provided educational material.
"""
        )

        human_message_prompt = HumanMessagePromptTemplate.from_template("{human_input}")

        return ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            human_message_prompt,
        ])

    def _advanced_code_validation(self, raw_code, educational_steps):
        """Advanced validation with enhanced error prevention and visual improvement"""
        
        # Extract and clean code
        code = self._extract_code_from_response(raw_code)
        
        # Apply comprehensive fixes
        code = self._apply_advanced_fixes(code, educational_steps)
        
        # Validate structure and syntax
        if not self._validate_advanced_structure(code):
            logging.warning("Advanced structure validation failed, using enhanced fallback")
            return self._get_comprehensive_fallback(educational_steps)
        
        # Final syntax validation
        try:
            compile(code, '<manim_scene>', 'exec')
            logging.info("Code passed syntax validation")
        except SyntaxError as e:
            logging.error(f"Syntax validation failed: {e}")
            return self._get_comprehensive_fallback(educational_steps)
        
        return code

    def _extract_code_from_response(self, raw_code):
        """Extract clean Python code from LLM response"""
        if '```python' in raw_code:
            code = raw_code.split('```python')[1].split('```')[0]
        elif '```' in raw_code:
            code = raw_code.split('```')[1]
        else:
            code = raw_code
        
        # Ensure proper imports
        if 'from manim import *' not in code:
            code = 'from manim import *\nimport numpy as np\n\n' + code
        elif 'import numpy as np' not in code:
            code = code.replace('from manim import *', 'from manim import *\nimport numpy as np')
        
        return code.strip()

    def _apply_advanced_fixes(self, code, educational_steps):
        """Apply advanced fixes for better visual quality and error prevention"""
        
        # Fix positioning with zone-based system
        code = self._apply_zone_positioning(code)
        
        # Enhance mathematical safety
        code = self._enhance_mathematical_safety(code)
        
        # Add smooth transitions
        code = self._add_smooth_transitions(code)
        
        # Fix common syntax issues
        code = self._fix_advanced_syntax_issues(code)
        
        # Add visual enhancements
        code = self._add_visual_enhancements(code)
        
        return code

    def _apply_zone_positioning(self, code):
        """Apply zone-based positioning to prevent overlaps"""
        zones = self.layout_zones
        
        # Title positioning
        code = re.sub(
            r'(\w+\s*=\s*Text\([^)]*title[^)]*\))',
            r'\1\n        \1.move_to(UP*3.2)\n        \1.set_z_index(10)',
            code, flags=re.IGNORECASE
        )
        
        # Step title positioning
        code = re.sub(
            r'(\w+\s*=\s*Text\([^)]*step[^)]*\))',
            r'\1\n        \1.move_to(UP*2.8)\n        \1.set_z_index(9)',
            code, flags=re.IGNORECASE
        )
        
        # Equation positioning with hierarchy
        code = re.sub(
            r'(\w+\s*=\s*MathTex\([^)]+\))',
            r'\1\n        \1.move_to(UP*1.2)\n        \1.set_z_index(8)',
            code
        )
        
        # Visual elements in center zone
        code = re.sub(
            r'(\w+\s*=\s*(?:Circle|Rectangle|Line|Arrow|Vector)\([^)]+\))',
            r'\1\n        \1.move_to(ORIGIN)\n        \1.set_z_index(5)',
            code
        )
        
        return code

    def _enhance_mathematical_safety(self, code):
        """Enhance mathematical expressions for safety and clarity"""
        
        # Safe function plotting with explicit ranges
        code = re.sub(
            r'\.plot\s*\(\s*lambda\s+(\w+):\s*([^,)]+)(?![^)]*x_range)',
            r'.plot(lambda \1: \2, x_range=[-4, 4]',
            code
        )
        
        # Wrap complex mathematical expressions
        unsafe_patterns = [
            (r'lambda\s+x:\s*x\*\*(\d+)', r'lambda x: x**\1 if abs(x) < 10 else 0'),
            (r'lambda\s+x:\s*(\d+)/x', r'lambda x: \1/x if abs(x) > 0.1 else 0'),
            (r'lambda\s+x:\s*np\.tan\(([^)]+)\)', r'lambda x: np.clip(np.tan(\1), -10, 10)'),
        ]
        
        for pattern, replacement in unsafe_patterns:
            code = re.sub(pattern, replacement, code)
        
        return code

    def _add_smooth_transitions(self, code):
        """Add smooth transitions between animation steps"""
        
        # Add transition methods if missing
        if 'def transition_to_next_step' not in code:
            transition_methods = '''
    def transition_to_next_step(self):
        """Smooth transition between educational steps"""
        current_objects = [obj for obj in self.mobjects if obj != self.axes]
        if current_objects:
            self.play(
                *[FadeOut(obj, shift=DOWN*0.3) for obj in current_objects],
                run_time=1.2
            )
            self.remove(*current_objects)
        self.wait(0.5)
    
    def highlight_concept(self, obj, color=YELLOW_E):
        """Highlight important concepts"""
        highlight = SurroundingRectangle(obj, color=color, buff=0.1)
        self.play(Create(highlight), run_time=1.0)
        self.wait(0.5)
        self.play(FadeOut(highlight), run_time=0.8)
        
    def emphasize_equation(self, equation):
        """Add emphasis to key equations"""
        self.play(Indicate(equation, scale_factor=1.2), run_time=1.5)
        self.wait(0.5)
'''
            # Insert after class definition
            code = code.replace(
                'def setup_educational_environment(self):',
                transition_methods + '\n    def setup_educational_environment(self):'
            )
        
        return code

    def _fix_advanced_syntax_issues(self, code):
        """Fix advanced syntax issues with better error handling"""
        
        # Fix incomplete method definitions
        code = re.sub(
            r'def\s+(\w+)\s*\([^)]*\):\s*$',
            r'def \1(self):\n        pass',
            code, flags=re.MULTILINE
        )
        
        # Fix missing self parameters
        code = re.sub(
            r'def\s+(\w+)\s*\(\s*\):',
            r'def \1(self):',
            code
        )
        
        # Fix incomplete play() calls
        code = re.sub(
            r'self\.play\(\s*\)',
            r'self.wait(1)',
            code
        )
        
        # Add font_size to Text objects without it (safer regex)
        code = re.sub(
            r'Text\(([^,)]+)\)(?!\s*,\s*[^,)]*font_size)',
            r'Text(\1, font_size=28)',
            code
        )
        
        return code

    def _add_visual_enhancements(self, code):
        """Add visual enhancements for better engagement"""
        
        # Add color specifications
        color_enhancements = [
            (r'Text\([^)]*title[^)]*\)(?![^,)]*color)', r'\g<0>, color=BLUE_E'),
            (r'Text\([^)]*step[^)]*\)(?![^,)]*color)', r'\g<0>, color=TEAL_E'),
            (r'MathTex\([^)]+\)(?![^,)]*color)', r'\g<0>, color=YELLOW_E'),
            (r'Circle\([^)]*\)(?![^,)]*color)', r'\g<0>, color=GREEN_E'),
            (r'Rectangle\([^)]*\)(?![^,)]*color)', r'\g<0>, color=BLUE_E'),
        ]
        
        for pattern, replacement in color_enhancements:
            code = re.sub(pattern, replacement, code)
        
        # Add proper timing
        if 'self.wait(' not in code:
            code = re.sub(
                r'(self\.play\([^)]+\))',
                r'\1\n        self.wait(1)',
                code
            )
        
        return code

    def _validate_advanced_structure(self, code):
        """Advanced validation for code structure and quality"""
        
        # Check for required methods
        required_methods = [
            'def construct(',
            'def setup_educational_environment(',
            'class ',
        ]
        
        for method in required_methods:
            if method not in code:
                logging.error(f"Missing required element: {method}")
                return False
        
        # Check for proper Manim usage
        manim_indicators = [
            'self.play(',
            'self.add(',
            'self.wait(',
            'Text(',
            'MathTex(',
        ]
        
        found_indicators = sum(1 for indicator in manim_indicators if indicator in code)
        if found_indicators < 3:
            logging.error(f"Insufficient Manim usage. Found: {found_indicators}")
            return False
        
        # Check for mathematical content
        math_indicators = ['MathTex', 'plot', 'lambda', 'np.', 'equation']
        if not any(indicator in code for indicator in math_indicators):
            logging.error("No mathematical content detected")
            return False
        
        return True

    def _get_comprehensive_fallback(self, educational_steps):
        """Generate a comprehensive fallback animation with proper structure"""
        
        step_count = len(educational_steps)
        class_name = f"MathAnimation{step_count}Steps"
        
        # Generate step methods
        step_methods = ""
        for i, step in enumerate(educational_steps[:5], 1):  # Limit to 5 steps
            step_title = step.get('step_title', f'Mathematical Step {i}')
            equations = step.get('equations', [])
            key_concepts = step.get('key_concepts', [])
            
            # Choose visualization type based on content
            if equations:
                visual_type = "equation"
                main_content = equations[0] if equations else "x^{}".format(i)
            elif any('function' in concept.lower() for concept in key_concepts):
                visual_type = "function"
                main_content = "sin({}x)".format(i)
            else:
                visual_type = "geometric"
                main_content = str(i)
            
            step_method = self._generate_step_method(i, step_title, visual_type, main_content)
            step_methods += step_method + "\n"
        
        return '''from manim import *
import numpy as np

class {}(Scene):'''.format(class_name) + '''
    def construct(self):
        self.setup_educational_environment()
        self.animate_step_by_step()
        self.create_final_summary()
    
    def setup_educational_environment(self):
        """Set up the 3Blue1Brown educational environment"""
        self.camera.background_color = "#0f0f23"
        
        # Professional coordinate system
        self.axes = Axes(
            x_range=[-6, 6, 1], 
            y_range=[-3.5, 3.5, 1],
            axis_config={{"color": BLUE_E, "stroke_width": 2, "stroke_opacity": 0.7}}
        )
        self.axes.set_opacity(0.4)
        self.add(self.axes)
        
        # Initialize tracking
        self.step_objects = []
        
    def clear_step(self):
        """Clear current step objects with smooth transition"""
        objects_to_clear = [obj for obj in self.mobjects if obj != self.axes]
        if objects_to_clear:
            self.play(*[FadeOut(obj, shift=DOWN*0.5) for obj in objects_to_clear], run_time=1.2)
            self.remove(*objects_to_clear)
        self.step_objects.clear()
        self.wait(0.5)
    
    def highlight_concept(self, obj):
        """Add emphasis to important concepts"""
        try:
            highlight = SurroundingRectangle(obj, color=YELLOW_E, buff=0.15)
            self.play(Create(highlight), run_time=1.0)
            self.wait(0.8)
            self.play(FadeOut(highlight), run_time=0.8)
        except:
            # Fallback if highlighting fails
            self.play(Indicate(obj), run_time=1.0)
            self.wait(0.5)
    
    def animate_step_by_step(self):
        """Execute all educational steps with proper transitions"""
        for step_num in range(1, min({} + 1, 6)):
            if step_num > 1:
                self.clear_step()
            method_name = "animate_step_" + str(step_num)
            if hasattr(self, method_name):
                getattr(self, method_name)()

{}
    
    def create_final_summary(self):
        """Create an engaging conclusion"""
        self.clear_step()
        
        # Title
        summary_title = Text("Mathematical Journey Complete", font_size=42, color=GOLD)
        summary_title.move_to(UP*2.5)
        
        # Key insights
        insights = VGroup(
            Text("✓ Explored fundamental concepts", font_size=28, color=GREEN_E),
            Text("✓ Visualized mathematical relationships", font_size=28, color=GREEN_E),
            Text("✓ Connected theory with intuition", font_size=28, color=GREEN_E)
        )
        insights.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        insights.move_to(ORIGIN)
        
        # Closing message
        closing = Text("Continue exploring mathematics...", font_size=24, color=TEAL_E, slant=ITALIC)
        closing.move_to(DOWN*2.5)
        
        # Animate summary
        self.play(Write(summary_title), run_time=2.5)
        self.wait(0.8)
        
        for insight in insights:
            self.play(Write(insight), run_time=1.5)
            self.wait(0.5)
        
        self.play(Write(closing), run_time=2.0)
        self.wait(2.0)
        
        # Final fade
        all_objects = [summary_title, insights, closing]
        self.play(*[FadeOut(obj) for obj in all_objects], run_time=2.5)
        self.wait(1.0)
'''.format(step_count, step_methods)

    def _generate_step_method(self, step_num, step_title, visual_type, content):
        """Generate a specific step method based on content type"""
        
        # Safely clean title and content to prevent string literal issues
        safe_title = ''.join(c for c in step_title[:30] if c.isalnum() or c in ' -_').strip()
        safe_content = ''.join(c for c in str(content)[:20] if c.isalnum() or c in '^+-*/')
        
        if visual_type == "equation":
            return '''    def animate_step_{}(self):
        """Step {}: Mathematical Concept"""
        # Step title
        title = Text("Step {}: {}", font_size=36, color=TEAL_E)
        title.move_to(UP*2.8)
        
        # Main equation
        equation = MathTex(r"{}", font_size=32, color=YELLOW_E)
        equation.move_to(UP*1.0)
        
        # Supporting visualization
        if hasattr(self.axes, 'plot'):
            try:
                func = self.axes.plot(lambda x: x**{} if abs(x) < 3 else 0, 
                                    color=BLUE_E, x_range=[-2, 2])
                
                # Animate sequence
                self.play(Write(title), run_time=2.0)
                self.wait(0.8)
                self.play(Write(equation), run_time=2.5)
                self.wait(1.0)
                self.play(Create(func), run_time=3.0)
                self.wait(1.5)
                
                # Highlight key concept
                self.highlight_concept(equation)
                
                self.step_objects.extend([title, equation, func])
            except:
                # Fallback without function plotting
                self.play(Write(title), run_time=2.0)
                self.wait(0.8)
                self.play(Write(equation), run_time=2.5)
                self.wait(1.5)
                self.step_objects.extend([title, equation])
        else:
            self.play(Write(title), run_time=2.0)
            self.wait(0.8)
            self.play(Write(equation), run_time=2.5)
            self.wait(1.5)
            self.step_objects.extend([title, equation])'''.format(step_num, step_num, step_num, safe_title, safe_content, step_num)
        
        elif visual_type == "function":
            return '''    def animate_step_{}(self):
        """Step {}: Mathematical Function"""
        # Step title
        title = Text("Step {}: {}", font_size=36, color=TEAL_E)
        title.move_to(UP*2.8)
        
        # Function visualization
        try:
            func = self.axes.plot(lambda x: np.sin({}*x) if abs(x) < 10 else 0, 
                                color=RED_E, x_range=[-3, 3], stroke_width=4)
            
            # Function label
            label = MathTex(r"f(x) = \\sin({}x)", font_size=28, color=YELLOW_E)
            label.move_to(UP*1.2)
            
            # Animate
            self.play(Write(title), run_time=2.0)
            self.wait(0.8)
            self.play(Write(label), run_time=2.0)
            self.wait(0.5)
            self.play(Create(func), run_time=3.5)
            self.wait(1.5)
            
            # Add key point
            key_point = Dot(self.axes.c2p(1, np.sin({})), color=GOLD, radius=0.08)
            self.play(DrawBorderThenFill(key_point), run_time=1.0)
            self.wait(1.0)
            
            self.step_objects.extend([title, func, label, key_point])
        except:
            # Fallback
            self.play(Write(title), run_time=2.0)
            self.wait(2.0)
            self.step_objects.append(title)'''.format(step_num, step_num, step_num, safe_title, step_num, step_num, step_num)
        
        else:  # geometric
            return '''    def animate_step_{}(self):
        """Step {}: Geometric Concept"""
        # Step title  
        title = Text("Step {}: {}", font_size=36, color=TEAL_E)
        title.move_to(UP*2.8)
        
        # Geometric shape
        radius = 0.8 + {} * 0.3
        shape = Circle(radius=radius, color=GREEN_E, stroke_width=3)
        shape.move_to(ORIGIN)
        
        # Mathematical property
        area_text = MathTex(r"A = \\pi r^2", font_size=24, color=WHITE)
        area_text.move_to(DOWN*2.0)
        
        # Animate
        self.play(Write(title), run_time=2.0)
        self.wait(0.8)
        self.play(Create(shape), run_time=2.5)
        self.wait(1.0)
        self.play(Write(area_text), run_time=2.0)
        self.wait(1.5)
        
        # Emphasis
        self.highlight_concept(shape)
        
        self.step_objects.extend([title, shape, area_text])'''.format(step_num, step_num, step_num, safe_title, step_num)

    def _extract_actual_content(self, educational_steps):
        """Extract actual mathematical/scientific content from educational steps"""
        content = {
            'equations': [],
            'key_concepts': [],
            'visual_elements': [],
            'real_world_examples': [],
            'step_descriptions': [],
            'animation_plans': [],
            'mathematical_operations': [],
            'scientific_phenomena': []
        }
        
        for step in educational_steps:
            # Extract equations
            equations = step.get('equations', [])
            if equations:
                content['equations'].extend(equations)
            
            # Extract key concepts
            key_concepts = step.get('key_concepts', [])
            if key_concepts:
                content['key_concepts'].extend(key_concepts)
            
            # Extract visual elements
            visual_elements = step.get('visual_elements', {})
            if visual_elements:
                content['visual_elements'].append({
                    'step': step.get('step_number', 0),
                    'elements': visual_elements
                })
            
            # Extract real-world examples
            examples = step.get('real_world_examples', [])
            if examples:
                content['real_world_examples'].extend(examples)
            
            # Extract descriptions for content analysis
            description = step.get('description', '')
            if description:
                content['step_descriptions'].append({
                    'step': step.get('step_number', 0),
                    'title': step.get('step_title', ''),
                    'description': description
                })
            
            # Extract animation plans
            animation_plan = step.get('animation_plan', '')
            if animation_plan:
                content['animation_plans'].append({
                    'step': step.get('step_number', 0),
                    'plan': animation_plan
                })
            
            # Analyze for mathematical operations
            if any(word in description.lower() for word in ['derivative', 'integral', 'function', 'graph', 'plot']):
                content['mathematical_operations'].append('calculus')
            if any(word in description.lower() for word in ['triangle', 'circle', 'polygon', 'geometric']):
                content['mathematical_operations'].append('geometry')
            if any(word in description.lower() for word in ['vector', 'matrix', 'linear']):
                content['mathematical_operations'].append('linear_algebra')
            
            # Analyze for scientific phenomena
            if any(word in description.lower() for word in ['wave', 'frequency', 'oscillation']):
                content['scientific_phenomena'].append('wave_physics')
            if any(word in description.lower() for word in ['force', 'motion', 'velocity', 'acceleration']):
                content['scientific_phenomena'].append('mechanics')
            if any(word in description.lower() for word in ['electron', 'atom', 'molecular']):
                content['scientific_phenomena'].append('atomic_physics')
        
        return content

    def _build_content_specific_instructions(self, educational_steps, content_analysis):
        """Build specific animation instructions based on actual content"""
        instructions = []
        
        # Add content overview
        instructions.append(f"CONTENT OVERVIEW:")
        instructions.append(f"- Total Steps: {len(educational_steps)}")
        instructions.append(f"- Equations to visualize: {len(content_analysis['equations'])}")
        instructions.append(f"- Key concepts: {', '.join(content_analysis['key_concepts'][:5])}")
        instructions.append(f"- Mathematical focus: {', '.join(set(content_analysis['mathematical_operations']))}")
        instructions.append(f"- Scientific phenomena: {', '.join(set(content_analysis['scientific_phenomena']))}")
        instructions.append("")
        
        # Add step-by-step specific instructions
        instructions.append("STEP-BY-STEP CONTENT REQUIREMENTS:")
        
        for i, step in enumerate(educational_steps, 1):
            step_title = step.get('step_title', f'Step {i}')
            description = step.get('description', '')
            key_concepts = step.get('key_concepts', [])
            equations = step.get('equations', [])
            visual_elements = step.get('visual_elements', {})
            animation_plan = step.get('animation_plan', '')
            narration = step.get('narration_script', '')
            
            instructions.append(f"STEP {i}: {step_title}")
            instructions.append(f"Description: {description[:200]}...")
            
            if key_concepts:
                instructions.append(f"Must visualize concepts: {', '.join(key_concepts)}")
            
            if equations:
                instructions.append(f"Must display equations: {', '.join(equations[:2])}")
            
            if visual_elements:
                diagrams = visual_elements.get('diagrams', [])
                animations = visual_elements.get('animations', [])
                if diagrams:
                    instructions.append(f"Required diagrams: {', '.join(diagrams)}")
                if animations:
                    instructions.append(f"Required animations: {', '.join(animations)}")
            
            if animation_plan:
                instructions.append(f"Specific animation plan: {animation_plan[:150]}...")
            
            if narration:
                instructions.append(f"Narration context: {narration[:100]}...")
            
            # Add content-specific visualization requirements
            self._add_visualization_requirements(instructions, step, content_analysis)
            
            instructions.append("")
        
        return "\n".join(instructions)

    def _add_visualization_requirements(self, instructions, step, content_analysis):
        """Add specific visualization requirements based on content type"""
        description = step.get('description', '').lower()
        key_concepts = [concept.lower() for concept in step.get('key_concepts', [])]
        
        # Mathematical visualization requirements
        if any(word in description for word in ['function', 'graph', 'plot']):
            instructions.append("MUST CREATE: Interactive function plot with axes and labeling")
        
        if any(word in description for word in ['triangle', 'circle', 'polygon']):
            instructions.append("MUST CREATE: Geometric shapes with proper measurements and labels")
        
        if any(word in description for word in ['derivative', 'slope', 'tangent']):
            instructions.append("MUST CREATE: Tangent line animation showing derivative concept")
        
        if any(word in description for word in ['integral', 'area']):
            instructions.append("MUST CREATE: Area under curve visualization with Riemann sums")
        
        if any(word in description for word in ['vector', 'direction']):
            instructions.append("MUST CREATE: Vector arrows with magnitude and direction")
        
        # Physics visualization requirements
        if any(word in description for word in ['wave', 'oscillation']):
            instructions.append("MUST CREATE: Animated wave propagation with frequency/amplitude controls")
        
        if any(word in description for word in ['force', 'motion']):
            instructions.append("MUST CREATE: Moving objects with force vectors and trajectories")
        
        if any(word in description for word in ['electric', 'magnetic', 'field']):
            instructions.append("MUST CREATE: Field line visualization with appropriate spacing")
        
        # Chemistry visualization requirements
        if any(word in description for word in ['molecule', 'atom', 'bond']):
            instructions.append("MUST CREATE: 3D molecular structure with bonds and electrons")
        
        if any(word in description for word in ['reaction', 'chemical']):
            instructions.append("MUST CREATE: Chemical equation with reactants and products animation")
        
        # General requirements based on equations
        equations = step.get('equations', [])
        if equations:
            instructions.append(f"MUST DISPLAY: All equations with proper LaTeX formatting: {', '.join(equations[:2])}")
            instructions.append("MUST ANIMATE: Step-by-step equation derivation or transformation")

# Initialize the enhanced generator
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
manim_generator = ManimGenerator(GROQ_API_KEY)