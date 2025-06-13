
**Prompt:**
Hi! I have a project idea I'd like help with. Here's an outline:

I want to build an **AI-powered math video generator**. The core concept is to combine a large language model (LLM) with the **Manim** framework to automatically create educational math videos.

### Key components:

1. **Script Generator LLM**:

   * This model takes a user’s prompt (e.g., “Explain the Pythagorean Theorem”) and generates a structured math explanation, including narration text and corresponding Manim animation instructions (in Python).

2. **Manim Instruction Model**:

   * This secondary LLM (or a specialized module) takes the generated script and converts it into optimized Manim code, ensuring smooth visuals and pedagogically sound explanations.

3. **Video Pipeline**:

   * The generated Manim code is rendered into an animated math video.

### Workflow:

* User enters a math-related prompt.
* The Script Generator LLM produces:

  * An explanation script.
  * Structured steps for visualization.
* The Manim LLM turns this into Manim code.
* The code is executed to produce a full educational video.

I'd like to develop this project for both stages and create a seamless pipeline. Please help me design the system, improve the prompt formatting for each stage, or suggest tools/models that can enhance the process.
