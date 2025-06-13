import os
from dotenv import load_dotenv
import logging
import json
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Basic logging configuration
logging.basicConfig(level=logging.INFO)

# Set comtypes logging level to WARNING to avoid informational messages
logging.getLogger('comtypes').setLevel(logging.WARNING)

load_dotenv('.env')

class ScriptGenerator:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
        self.groq_chat = ChatGroq(groq_api_key=self.groq_api_key, model_name='llama3-8b-8192')
        self.prompt = self._create_chat_prompt()
        self.conversation = ConversationChain(
            llm=self.groq_chat,
            prompt=self.prompt,
            verbose=True,
            memory=self.memory,
            input_key="human_input",
        )

    def generate_script(self, prompt):
        """
        Generate a structured math educational script based on the user's query.

        Args:
            prompt (str): User's mathematical topic query.

        Returns:
            dict: Structured educational content or empty dict if error.
        """
        if not prompt:
            return {}
        
        try:
            # Create a math-specific prompt
            math_prompt = f"""
            Create a structured educational explanation for the mathematical topic: "{prompt}"
            
            Please provide a comprehensive breakdown suitable for an educational video that will help students understand this concept clearly.
            
            Return your response as a valid JSON object following the specified structure.
            """
            
            response = self.conversation.predict(human_input=math_prompt)
            
            # Try to parse the JSON response
            try:
                structured_content = json.loads(response)
                return structured_content
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the response
                structured_content = self._extract_json_from_response(response)
                if structured_content:
                    return structured_content
                else:
                    # Fallback: create a basic structure
                    return self._create_fallback_structure(prompt, response)
                    
        except Exception as e:
            print(f"Error processing request: {e}")
            return {}

    def _extract_json_from_response(self, response):
        """
        Try to extract JSON from a response that might contain additional text.
        
        Args:
            response (str): The LLM response
            
        Returns:
            dict: Parsed JSON or None if extraction fails
        """
        try:
            # Look for JSON content between ```json and ``` or { and }
            import re
            
            # Try to find JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
              # Try to find JSON object
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
                
        except Exception as e:
            print(f"JSON extraction failed: {e}")
            
        return None
    
    def _create_fallback_structure(self, prompt, response):
        """
        Create a basic structure when JSON parsing fails.
        
        Args:
            prompt (str): Original prompt
            response (str): LLM response
            
        Returns:
            dict: Basic structured content
        """
        # Extract topic from enhanced prompt if it exists
        topic = self._extract_topic_from_prompt(prompt)
        
        return {
            "title": topic,
            "introduction": f"An explanation of {topic}",
            "explanation_steps": [
                {
                    "step": 1,
                    "narration": response[:500] + "...",
                    "visual_description": "Display the mathematical concept with appropriate notation",
                    "mathematical_objects": ["text", "equation"],
                    "duration": 10,
                    "key_equation": "",
                    "emphasis_points": [topic]
                }
            ],
            "summary": "This completes our explanation of the concept.",
            "complexity_level": "intermediate",
            "mathematical_domain": "general",
            "total_duration": 30
        }
    
    def _extract_topic_from_prompt(self, prompt):
        """
        Extract the actual topic from an enhanced prompt.
        
        Args:
            prompt (str): The original prompt (may be enhanced)
            
        Returns:
            str: The extracted topic
        """
        import re
        
        # Look for the pattern "Create an educational explanation for: [topic]"
        match = re.search(r'Create an educational explanation for:\s*([^\n]+)', prompt)
        if match:
            topic = match.group(1).strip()
            # Clean up the topic
            topic = re.sub(r'\s+', ' ', topic)
            if len(topic) > 50:
                topic = topic[:47] + "..."
            return topic
        
        # Fallback: use first 50 characters of prompt
        clean_prompt = prompt.replace('\n', ' ').strip()
        clean_prompt = re.sub(r'\s+', ' ', clean_prompt)
        if len(clean_prompt) > 50:
            return clean_prompt[:47] + "..."
        return clean_prompt

    def generate_raw_script(self, prompt):
        """
        Generate a raw script (backward compatibility).
        
        Args:
            prompt (str): User's query.

        Returns:
            str: Generated script text.
        """
        structured_content = self.generate_script(prompt)
        if not structured_content:
            return ""
        
        # Convert structured content to readable script
        script_parts = []
        script_parts.append(f"Title: {structured_content.get('title', '')}")
        script_parts.append(f"Introduction: {structured_content.get('introduction', '')}")
        
        for step in structured_content.get('explanation_steps', []):
            script_parts.append(f"Step {step.get('step', '')}: {step.get('narration', '')}")
        
        script_parts.append(f"Summary: {structured_content.get('summary', '')}")
        
        return "\n\n".join(script_parts)

    def _create_chat_prompt(self):
        """
        Create a math-specific chat prompt template for educational content generation.

        Returns:
            ChatPromptTemplate: The chat prompt template.
        """
        system_message = SystemMessage(
            content='''
            You are an expert mathematics educator and content creator specializing in generating structured educational content for math video production.

            Your task is to create comprehensive, pedagogically sound mathematical explanations that will be used to generate educational videos using the Manim animation framework.

            When given a mathematical topic or concept, you must produce a JSON response with the following structure:

            {
                "title": "Clear, concise title of the concept",
                "introduction": "Brief overview of what will be covered",
                "explanation_steps": [
                    {
                        "step": 1,
                        "narration": "What the narrator should say (natural, conversational tone)",
                        "visual_description": "Detailed description of what should be animated (geometric shapes, equations, transformations, etc.)",
                        "mathematical_objects": ["list", "of", "math", "objects", "needed"],
                        "duration": 5,
                        "key_equation": "LaTeX format equation if applicable",
                        "emphasis_points": ["key", "concepts", "to", "highlight"]
                    }
                ],
                "summary": "Brief recap of key takeaways",
                "complexity_level": "beginner|intermediate|advanced",
                "mathematical_domain": "algebra|geometry|calculus|statistics|etc",
                "total_duration": 60
            }

            Guidelines:
            1. Break down complex concepts into digestible steps (3-8 steps typically)
            2. Use clear, engaging narration suitable for educational videos
            3. Provide specific visual descriptions that can be translated to Manim code
            4. Include proper mathematical notation in LaTeX format
            5. Ensure logical progression from basic to advanced concepts
            6. Suggest appropriate timing for each step
            7. Identify key mathematical objects needed for visualization
            8. Maintain pedagogical soundness and mathematical accuracy

            Focus on creating content that is both educationally effective and visually engaging.
            '''
        )

        human_message_prompt = HumanMessagePromptTemplate.from_template("{human_input}")

        prompt = ChatPromptTemplate.from_messages(
            [
                system_message,
                MessagesPlaceholder(variable_name="chat_history"),
                human_message_prompt,
            ]
        )
        return prompt

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize the script generator
script_generator = ScriptGenerator(GROQ_API_KEY)
