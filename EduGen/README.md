# 🎓 EduGen: AI-Powered Math Video Generator

An intelligent system that combines Large Language Models (LLMs) with the Manim animation framework to automatically create educational mathematics videos.

## 🚀 Features

### ✨ Two-Stage AI Pipeline
1. **Educational Content Generator**: Creates structured, pedagogically sound mathematical explanations
2. **Manim Code Generator**: Converts educational content into executable animation code

### 🎯 Key Capabilities
- **Math-Specific Content Generation**: Tailored prompts for different mathematical domains
- **Structured Output**: JSON-formatted content with visual descriptions and timing
- **Professional Animations**: Manim-powered mathematical visualizations
- **Complete Animation Pipeline**: Automated generation and rendering of mathematical animations

### 🧮 Supported Mathematical Domains
- Algebra
- Geometry  
- Calculus
- Statistics
- Trigonometry
- General Mathematics

## 🏗️ Architecture

```
User Input → Content Generator → Manim Code Generator → Animation Renderer → Final Animation
```

### Stage 1: Educational Content Generator
- **Input**: Mathematical topic (e.g., "Pythagorean Theorem")
- **Output**: Structured JSON with:
  - Step-by-step explanations
  - Visual descriptions
  - Mathematical objects needed
  - Timing information
  - LaTeX equations

### Stage 2: Manim Code Generator  
- **Input**: Structured educational content
- **Output**: Complete, executable Manim Python code
- **Features**: Mathematical objects, animations, proper timing, visual appeal

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   cd "/media/nafiz/NewVolume/Python Mini Projects/EduGen"
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies**
   ```bash
   # LaTeX (for mathematical notation)
   sudo apt install texlive-full  # Ubuntu/Debian
   # brew install --cask mactex    # macOS
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

## 🔧 Configuration

Create a `.env` file with your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional fallback
```

## 🎮 Usage

### Web Interface (Streamlit)
```bash
streamlit run streamlit_app.py
```

Then navigate to `http://localhost:8501` and:
1. Enter a mathematical topic
2. Select complexity level and domain
3. Generate structured content
4. Generate Manim animation code
5. Render and download animation

### Programmatic Usage

```python
from script_generator import script_generator
from manim_code_generator import manim_code_generator
from animation_creator import create_animation_from_code

# Generate structured content
content = script_generator.generate_script("Explain quadratic equations")

# Generate Manim code
manim_code = manim_code_generator.generate_manim_code(content)

# Create animation
video_path = create_animation_from_code(manim_code)
```

## 📁 Project Structure

```
EduGen/
├── script_generator.py          # Stage 1: Educational content generation
├── manim_code_generator.py      # Stage 2: Manim code generation
├── animation_creator.py         # Animation rendering
├── streamlit_app.py             # Web interface
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
└── media/                       # Generated content storage
```

## 🎨 Example Output Structure

### Generated Content (Stage 1)
```json
{
  "title": "The Pythagorean Theorem",
  "introduction": "Let's explore one of the most famous theorems in mathematics...",
  "explanation_steps": [
    {
      "step": 1,
      "narration": "We begin with a right triangle with sides a, b, and hypotenuse c",
      "visual_description": "Create a right triangle with labeled sides",
      "mathematical_objects": ["right_triangle", "labels", "text"],
      "duration": 5,
      "key_equation": "a^2 + b^2 = c^2",
      "emphasis_points": ["right triangle", "hypotenuse"]
    }
  ],
  "complexity_level": "intermediate",
  "mathematical_domain": "geometry",
  "total_duration": 60
}
```

### Generated Manim Code (Stage 2)
```python
from manim import *

class PythagoreanTheoremScene(Scene):
    def construct(self):
        # Create right triangle
        triangle = RightTriangle()
        labels = VGroup(
            MathTex("a").next_to(triangle.get_left()),
            MathTex("b").next_to(triangle.get_bottom()),
            MathTex("c").next_to(triangle.get_hypotenuse())
        )
        
        self.play(Create(triangle), run_time=2)
        self.play(Write(labels), run_time=2)
        self.wait(2)
        
        # Show the theorem
        theorem = MathTex("a^2 + b^2 = c^2")
        self.play(Write(theorem))
        self.wait(3)
```

## 🚧 Development Status

### Phase 1 ✅ (Completed)
- [x] Enhanced script generator with math-specific prompts
- [x] Two-stage LLM pipeline
- [x] Manim code generator
- [x] Improved Streamlit interface

### Next Steps 📋
- [ ] Mathematical object library
- [ ] Advanced error handling
- [ ] Code validation system
- [ ] Animation preview

---

*Built with ❤️ for mathematics education*