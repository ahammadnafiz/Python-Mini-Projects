# InnoGenie 2.0 🧞‍♂️💡

InnoGenie 2.0 is an AI-powered idea generation and exploration tool designed to spark innovation across various industries. Leveraging the power of LLaMA3 70b, Langchain, and the Groq API, this application helps users generate, analyze, and refine innovative ideas.

## Features

- **Idea Generation**: Generate 5 innovative, practical, and implementable ideas based on user-defined parameters.
- **Idea Exploration**: Dive deeper into generated ideas with an AI-powered Q&A system.
- **Inspiration Prompts**: Get creative inspiration to kickstart your ideation process.
- **Customizable Creativity**: Adjust the AI's creativity level to fine-tune idea generation.
- **Idea Management**: Save and load generated ideas for later use.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/ahammadnafiz/Python-Mini-Projects.git
   cd InnoGenie
   ```

2. Set up your environment variables:
   - Create a `.env` file in the root directory
   - Add your Groq API key: `GROQ_API_KEY=your_api_key_here`

## Usage

1. Run the Streamlit app:
   ```
   streamlit innogenie.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter your Groq API key in the sidebar (or it will be loaded from your `.env` file).

4. Use the interface to generate and explore innovative ideas:
   - Fill in the "Area of Interest," "Tags," and "Subcategory" fields
   - Click "Generate Ideas" to create new ideas
   - Use the "Get Inspiration" button for creative prompts
   - Explore generated ideas using the built-in chat interface

## Configuration

- Adjust the "Creativity Level" slider in the sidebar to control the AI's temperature setting.
- Toggle "Show Debug Info" in the sidebar to view session state and debugging information.

## Dependencies

- streamlit
- langchain
- langchain_groq
- pydantic
- python-dotenv

## Contributing

Contributions to InnoGenie 2.0 are welcome! Please feel free to submit pull requests, create issues, or suggest new features.

## Support

For support, please open an issue on the GitHub repository or contact the maintainers directly.

---

Happy Innovating with InnoGenie 2.0! 🚀✨