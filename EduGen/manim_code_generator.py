import os
import re
import json
import logging
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logging.getLogger('comtypes').setLevel(logging.WARNING)

load_dotenv('.env')

class ManIMCodeGenerator:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
        self.groq_chat = ChatGroq(groq_api_key=self.groq_api_key, model_name='llama-3.3-70b-versatile')
        
        # Enhanced Manim code generation prompt
        self.manim_prompt = self._create_manim_generation_prompt()
        
        # Manim conversation chain
        self.manim_conversation = ConversationChain(
            llm=self.groq_chat,
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
            print(f"📚 Topic: {educational_breakdown.get('title', 'Unknown')}")
            print(f"🎯 Educational Steps: {len(educational_breakdown.get('educational_steps', []))}")
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
                print(f"📝 Code Length: {len(manim_code)} characters")
                print("🎬 Ready for animation rendering!")
                
                # Display generated manim code in terminal
                self._display_manim_code(manim_code)
                
                return manim_code
            else:
                raise Exception("Code extraction failed")
                
        except Exception as e:
            print(f"❌ Error in Manim code generation: {e}")
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
        
        prompt_parts = [f"""
ADVANCED MANIM CODE GENERATION REQUEST

VIDEO PLAN TO IMPLEMENT:
Title: {title}
Duration: {duration} seconds
Educational Steps: {len(steps)}

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

EDUCATIONAL STEPS TO IMPLEMENT:"""]
        
        # Add detailed information about each educational step
        for i, step in enumerate(steps, 1):
            prompt_parts.append(f"""

Step {i}: {step.get('step_title', f'Step {i}')}
- Duration: {step.get('duration_seconds', 30)} seconds
- Key Concepts: {', '.join(step.get('key_concepts', []))}
- Narration: {step.get('narration_script', '')}
- Visual Plan: {step.get('animation_plan', '')}
- Visual Elements: {step.get('visual_elements', {})}
- Equations: {step.get('equations', [])}
- Real-world Examples: {step.get('real_world_examples', [])}""")

        prompt_parts.append(f"""

TOTAL DURATION: {duration} seconds
TARGET COMPLEXITY: {educational_breakdown.get('metadata', {}).get('difficulty_progression', 'intermediate')}

OUTPUT FORMAT:
Provide complete, executable Manim Python code following this structure:

```python
from manim import *

class {title.replace(' ', '').replace(':', '')}Scene(Scene):
    def construct(self):
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
        title = Text("{title}", font_size=48, color=BLUE).shift(UP*3)
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
            - Text("text",.shift() - missing closing parenthesis before comma
            - Text("text"), .shift() - comma before method call
            - Text("text" .shift() - missing closing parenthesis and quote
            - Missing closing parentheses in method calls
            
            CORRECT SYNTAX EXAMPLES:
            - Text("Hello World").shift(UP*2) ✅
            - Text("Hello", font_size=24).shift(DOWN*1) ✅
            - Text("Title", color=BLUE).shift(UP*3).scale(0.8) ✅
            
            FORBIDDEN SYNTAX EXAMPLES:
            - Text("Hello",.shift(UP*2) ❌ (comma before method)
            - Text("Hello").shift(UP*2 ❌ (missing closing parenthesis)
            - Text("Hello World",.shift(UP*2) ❌ (comma + missing closing paren)
            - Text("Hello World", .shift(UP*2) ❌ (comma space before method)
            
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
        print(f"🎯 Title: {educational_breakdown.get('title', 'N/A')}")
        print(f"📝 Abstract: {educational_breakdown.get('abstract', 'N/A')[:100]}...")
        print(f"⏱️  Duration: {educational_breakdown.get('metadata', {}).get('estimated_total_duration', 'N/A')} seconds")
        print(f"👥 Target Audience: {educational_breakdown.get('metadata', {}).get('target_audience', 'N/A')}")
        
        # Learning objectives
        objectives = educational_breakdown.get('learning_objectives', [])
        if objectives:
            print(f"\n🎯 Learning Objectives ({len(objectives)}):")
            for i, obj in enumerate(objectives[:3], 1):
                print(f"   {i}. {obj}")
            if len(objectives) > 3:
                print(f"   ... and {len(objectives) - 3} more")
        
        # Educational steps
        steps = educational_breakdown.get('educational_steps', [])
        if steps:
            print(f"\n📚 Educational Steps ({len(steps)}):")
            for i, step in enumerate(steps, 1):
                title = step.get('step_title', f'Step {i}')
                duration = step.get('duration_seconds', 'N/A')
                concepts = step.get('key_concepts', [])
                print(f"   {i}. {title} ({duration}s)")
                if concepts:
                    print(f"      Key Concepts: {', '.join(concepts[:3])}")
        
        # Manim structure
        if manim_structure:
            animation_steps = manim_structure.get('animation_steps', [])
            print(f"\n🎬 Animation Steps ({len(animation_steps)}):")
            for i, step in enumerate(animation_steps[:3], 1):
                objects = step.get('manim_objects', [])
                animations = step.get('animations', [])
                print(f"   {i}. {step.get('description', 'Animation step')}")
                print(f"      Objects: {', '.join(objects[:3])}")
                print(f"      Animations: {', '.join(animations[:3])}")
            if len(animation_steps) > 3:
                print(f"   ... and {len(animation_steps) - 3} more steps")
        
        # Generation metadata
        if generation_metadata:
            print(f"\n⚙️  Generation Info:")
            print(f"   Stages Completed: {generation_metadata.get('stages_completed', [])}")
            print(f"   Total Duration: {generation_metadata.get('total_duration', 'N/A')} seconds")
            print(f"   Complexity: {generation_metadata.get('complexity_level', 'N/A')}")
        
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
        print(f"📊 Code Statistics:")
        print(f"   Lines of code: {len(lines)}")
        print(f"   Characters: {len(manim_code)}")
        
        # Check for key components
        has_imports = 'from manim import' in manim_code or 'import manim' in manim_code
        has_class = 'class' in manim_code and 'Scene' in manim_code
        has_construct = 'def construct' in manim_code
        has_methods = manim_code.count('def ') > 1
        
        print(f"   Has imports: {'✅' if has_imports else '❌'}")
        print(f"   Has scene class: {'✅' if has_class else '❌'}")
        print(f"   Has construct method: {'✅' if has_construct else '❌'}")
        print(f"   Has additional methods: {'✅' if has_methods else '❌'}")
        
        # Extract class name
        import re
        class_match = re.search(r'class\s+(\w+)', manim_code)
        if class_match:
            print(f"   Class name: {class_match.group(1)}")
        
        # Show first few lines and last few lines
        print(f"\n📝 Code Preview:")
        print("─" * 40)
        
        # First 15 lines
        for i, line in enumerate(lines[:15], 1):
            print(f"{i:2}: {line}")
        
        if len(lines) > 30:
            print("   ...")
            print(f"   [... {len(lines) - 30} lines omitted ...]")
            print("   ...")
            
            # Last 15 lines
            for i, line in enumerate(lines[-15:], len(lines) - 14):
                print(f"{i:2}: {line}")
        elif len(lines) > 15:
            # Show remaining lines if total is between 15-30
            for i, line in enumerate(lines[15:], 16):
                print(f"{i:2}: {line}")
        
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
        if not code:
            return code
        
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
                fixed_lines.append(f"        # REMOVED: {line.strip()} (set_background method doesn't exist in Manim)")
                fixed_lines.append(f"        # Background color is set in Manim config or using Camera background_color")
                fixed_lines.append(f"        # Scene backgrounds are handled automatically by Manim")
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
                            line = line.rstrip() + f'.shift({pos})'
                            text_positions_used.add(pos)
                            break
            
            # Check for missing scene clearing between methods
            if 'def ' in line and 'construct' not in line and '__init__' not in line:
                # This is a new method - ensure it starts with clearing if it's not the first method
                method_name = line.strip()
                fixed_lines.append(line)
                # Add clearing instruction as comment
                fixed_lines.append(f"        # Clear previous content to avoid overlap")
                fixed_lines.append(f"        # self.play(FadeOut(*self.mobjects)) # Uncomment if needed")
                continue
            
            # Skip lines with ImageMobject references
            if 'ImageMobject' in line or 'Image.open' in line:
                # Replace with a comment explaining what was removed
                comment_line = line.strip()
                fixed_lines.append(f"        # REMOVED: {comment_line} (ImageMobject not supported)")
                fixed_lines.append(f"        # Using text description instead:")
                
                # Extract variable name if possible
                if '=' in line and 'ImageMobject' in line:
                    var_name = line.split('=')[0].strip()
                    # Replace with a text description with positioning
                    fixed_lines.append(f"        {var_name} = Text('Visual representation of concept', font_size=24).shift(DOWN*1)")
                continue
            
            # Skip lines that reference image files
            if any(ext in line.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.ico']):
                # Comment out the problematic line
                fixed_lines.append(f"        # REMOVED: {line.strip()} (Image file reference not supported)")
                continue
            
            # Fix other common Manim method errors
            if any(method in line for method in ['self.set_color_scheme', 'self.set_theme', 'self.configure_camera']):
                # Comment out invalid methods
                fixed_lines.append(f"        # REMOVED: {line.strip()} (Invalid Manim method)")
                continue
                
            # Fix common import issues
            if line.strip() == 'from manim import *':
                fixed_lines.append(line)
            elif 'import' in line and any(img_ref in line for img_ref in ['PIL', 'Image', 'cv2', 'opencv']):
                # Comment out image-related imports
                fixed_lines.append(f"        # REMOVED: {line.strip()} (Image library import not needed)")
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
            print(f"❌ Final syntax error detected: {e.msg}")
            print(f"   Problem line {e.lineno}: {e.text}")
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
        
        # Pre-process to fix the most common pattern that causes issues
        # Fix the exact pattern: Text("text",.shift(UP*1) -> Text("text").shift(UP*1)
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            original_line = line
            
            # Look for the specific problematic pattern: Text("text",.shift(UP*1)
            if 'Text(' in line and '",' in line and '.shift(' in line:
                # Pattern: Text("some text",.shift(UP*1)
                # Fix by replacing ",.shift( with ").shift(
                line = re.sub(r'Text\("([^"]*)",\s*\.shift\(', r'Text("\1").shift(', line)
                print(f"🔧 Pre-fixed Text+comma+shift pattern: {line.strip()}")
            
            # Fix pattern: Text("text"), .shift(
            if 'Text(' in line and '"), .' in line:
                line = re.sub(r'Text\("([^"]*)"\),\s*\.', r'Text("\1").', line)
                print(f"🔧 Pre-fixed Text+closing+comma+dot pattern: {line.strip()}")
            
            # Fix any Text objects with missing closing parenthesis before .shift
            if 'Text(' in line and '.shift(' in line:
                # Check if there's an unmatched quote-comma pattern
                if '",.' in line or '",.shift(' in line:
                    line = line.replace('",', '")')
                    print(f"🔧 Pre-fixed quote-comma pattern: {line.strip()}")
            
            # Handle missing closing parentheses
            if 'Text(' in line and '.shift(' in line and line.count('(') > line.count(')'):
                # Count missing closing parentheses
                missing_parens = line.count('(') - line.count(')')
                line = line.rstrip() + ')' * missing_parens
                print(f"🔧 Added {missing_parens} missing closing parenthesis(es): {line.strip()}")
            
            # Fix any remaining ",.method(" patterns
            line = re.sub(r'",\s*\.(shift|scale|rotate|set_color)\(', r'").\1(', line)
            
            if line != original_line:
                print(f"   Original: {original_line.strip()}")
                print(f"   Fixed:    {line.strip()}")
            
            fixed_lines.append(line)
        
        code = '\n'.join(fixed_lines)
        
        # Now apply the regular fixes
        fixes = [
            # Fix the specific error: Text("text",.shift() -> Text("text").shift()
            (r'Text\([^)]+\),\s*\.shift\(', lambda m: m.group(0).replace('",', '")')),
            (r'Text\([^)]+\)\s*,\s*\.shift\(', lambda m: m.group(0).replace(',', '')),
            
            # Fix missing closing parentheses with extra comma before method calls
            (r'Text\([^)]+\),\.(shift|scale|rotate|set_color)', lambda m: m.group(0).replace(',', ')')),
            
            # Fix specific pattern: "text",.shift -> "text").shift
            (r'"[^"]*",\s*\.(shift|scale|rotate)', r'").\1'),
            
            # Fix double commas
            (r',,', ','),
            
            # Fix trailing commas before method calls
            (r',\s*\.(shift|scale|rotate|set_color)', r'.\1'),
            
            # Fix missing quotes around strings
            (r'Text\(\s*([^"\'][^,)]*[^"\'])\s*\)', r'Text("\1")'),
            
            # Fix malformed method chaining
            (r'\.shift\([^)]*\)\.(shift|scale|rotate)', r'.shift(UP*1).\1'),
            
            # Fix missing parentheses in Text() calls
            (r'Text\s*\(\s*([^)]+)\s*\.', r'Text(\1).'),
            
            # Fix missing closing parentheses at end of lines
            (r'Text\([^)]*$', lambda m: m.group(0) + ')'),
            
            # Fix spacing issues around method calls
            (r'\s+\.', '.'),
            
            # Fix invalid color references
            (r'color=([A-Z_]+)([^A-Z_,)])', r'color=\1, \2'),
            
            # Fix specific pattern with parentheses mismatch
            (r'Text\("([^"]*)",\s*\.', r'Text("\1").'),
            
            # Fix any remaining malformed Text calls
            (r'Text\("([^"]*)",[^)]*\.shift\(([^)]*)\)', r'Text("\1").shift(\2)'),
        ]
        
        fixed_code = code
        
        for pattern, replacement in fixes:
            if callable(replacement):
                # Use re.sub with a function
                fixed_code = re.sub(pattern, replacement, fixed_code)
            else:
                # Use simple string replacement
                fixed_code = re.sub(pattern, replacement, fixed_code)
        
        # Check for syntax errors by trying to parse
        try:
            import ast
            ast.parse(fixed_code)
            print("✅ Syntax validation passed")
        except SyntaxError as e:
            print(f"⚠️ Syntax error detected at line {e.lineno}: {e.msg}")
            print(f"   Problem line: {e.text}")
            
            # Try to fix the specific error
            if e.lineno and e.text:
                lines = fixed_code.split('\n')
                if e.lineno <= len(lines):
                    problem_line = lines[e.lineno - 1]
                    
                    # Common fixes for specific syntax errors
                    if 'invalid syntax' in e.msg.lower():
                        # Fix common parentheses issues
                        if '.shift(' in problem_line and problem_line.count('(') > problem_line.count(')'):
                            # Missing closing parenthesis
                            fixed_line = problem_line + ')'
                            lines[e.lineno - 1] = fixed_line
                            print(f"🔧 Fixed missing closing parenthesis: {fixed_line.strip()}")
                        
                        elif ',.' in problem_line or ',.shift(' in problem_line:
                            # Extra comma before dot or method call
                            fixed_line = problem_line.replace(',.', '.').replace(',', ')')
                            lines[e.lineno - 1] = fixed_line
                            print(f"🔧 Fixed extra comma: {fixed_line.strip()}")
                        
                        elif 'Text(' in problem_line and problem_line.count('"') % 2 == 1:
                            # Missing quote
                            if not problem_line.rstrip().endswith('"'):
                                fixed_line = problem_line.rstrip() + '")'
                            else:
                                fixed_line = problem_line + '"'
                            lines[e.lineno - 1] = fixed_line
                            print(f"🔧 Fixed missing quote: {fixed_line.strip()}")
                        
                        elif 'Text(' in problem_line and '),' in problem_line and '.shift(' in problem_line:
                            # Text("text",.shift( pattern - remove comma after closing paren
                            fixed_line = problem_line.replace('),', ')')
                            lines[e.lineno - 1] = fixed_line
                            print(f"🔧 Fixed comma after closing parenthesis: {fixed_line.strip()}")
                    
                    fixed_code = '\n'.join(lines)
                    
                    # Try parsing again
                    try:
                        ast.parse(fixed_code)
                        print("✅ Syntax error fixed successfully")
                    except SyntaxError as e2:
                        print(f"❌ Could not automatically fix syntax error: {e2.msg}")
                        # Return original code with comment about the error
                        fixed_code = f"# SYNTAX ERROR DETECTED: {e.msg}\n# LINE {e.lineno}: {e.text}\n\n" + fixed_code
        
        return fixed_code

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
        print(f"🚨 Emergency syntax fix for line {syntax_error.lineno}: {problem_line.strip()}")
        
        # Common emergency fixes
        fixed_line = problem_line
        
        # Fix the specific pattern we keep seeing
        if 'Text(' in problem_line and '",' in problem_line and '.shift(' in problem_line:
            # Pattern: Text("text",.shift(UP*1) -> Text("text").shift(UP*1)
            fixed_line = re.sub(r'Text\("([^"]*)",\s*\.shift\(([^)]*)\)', r'Text("\1").shift(\2)', problem_line)
            print(f"🔧 Emergency fix applied: {fixed_line.strip()}")
        
        # Fix missing closing parentheses
        elif problem_line.count('(') > problem_line.count(')'):
            missing = problem_line.count('(') - problem_line.count(')')
            fixed_line = problem_line.rstrip() + ')' * missing
            print(f"🔧 Emergency fix: added {missing} closing parentheses")
        
        # Fix missing opening parentheses (rare)
        elif problem_line.count(')') > problem_line.count('('):
            # Find where to add opening parentheses (usually after =)
            if '=' in problem_line and 'Text(' in problem_line:
                parts = problem_line.split('=', 1)
                if len(parts) == 2:
                    fixed_line = parts[0] + '= Text(' + parts[1].lstrip().lstrip('Text(')
                    print(f"🔧 Emergency fix: added opening parenthesis")
        
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
                        print(f"🔧 Emergency fix: added missing quote")
        
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
                print(f"❌ Emergency fix failed: {e2.msg}")
                # Return original code with a comment about the error
                return f"# SYNTAX ERROR DETECTED: {syntax_error.msg}\n# LINE {syntax_error.lineno}: {syntax_error.text}\n\n" + code
        
        # If no fix was applied, return original with error comment
        return f"# SYNTAX ERROR DETECTED: {syntax_error.msg}\n# LINE {syntax_error.lineno}: {syntax_error.text}\n\n" + code

# Initialize the Manim code generator
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if GROQ_API_KEY:
    manim_generator = ManIMCodeGenerator(GROQ_API_KEY)
    print("✅ Advanced Manim Code Generator initialized successfully!")
    print("🎨 Ready to generate dynamic, educational animations!")
else:
    print("❌ GROQ_API_KEY not found. Please set your API key in the .env file.")