# ConvertirseAI: Advanced Code Conversion Platform

ConvertirseAI is an advanced code conversion platform powered by LLaMA3 70b, Langchain, and Groq API. It leverages state-of-the-art AI to transform code between various programming languages while maintaining code structure and functionality.

## Features

- **Code Conversion**: Convert code snippets from one programming language to another using advanced AI models.
- **Syntax Preservation**: Preserve comments and structure during code conversion.
- **Language Support**: Currently supports Python, JavaScript, Java, C++, Ruby, Go, Rust, TypeScript, PHP, and Swift.
- **Explanation**: Understand the transformation process with an explanation of key differences between source and target code.
- **Feedback**: Provide feedback to improve the platform.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/convertirse-ai.git
   ```

2. Navigate to the project directory:

   ```
   cd ConvertirseAI
   ```


## Usage

1. Set up your environment with the necessary API keys:

   - Obtain your Groq API key and set it as an environment variable:

     ```
     export GROQ_API_KEY="your_groq_api_key_here"
     ```

2. Run the Streamlit app:

   ```
   streamlit run convertirse.py
   ```

3. Use the web interface to:

   - Select the source and target programming languages.
   - Paste your source code snippet in the provided text area.
   - Click on "Transform Code" to convert the code to the target language.
   - Explore the transformed code and its explanation.

4. Provide feedback to help us improve the platform.

## Contributing

Contributions are welcome! If you'd like to contribute to ConvertirseAI, please follow these steps:

1. Fork the repository and create your branch:

   ```
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:

   ```
   git commit -m 'Add your feature or fix'
   ```

3. Push to your branch:

   ```
   git push origin feature/your-feature-name
   ```

4. Create a new Pull Request.


## Acknowledgments

- Streamlit: For providing an excellent framework for building interactive web applications with Python.
- Langchain and Groq: For their powerful tools and APIs that enable advanced code conversion capabilities.

---

### Notes:
- Customize the URLs and placeholders (`your-username`, `your_groq_api_key_here`, etc.) with your actual information.
- Ensure to include any additional setup instructions or prerequisites specific to your environment or deployment scenario.
- Update the acknowledgments section to include any other third-party libraries or tools used in your project.
