import os
from dotenv import load_dotenv
import logging
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
        Generate a script based on the user's query.

        Args:
            prompt (str): User's query.

        Returns:
            str: Generated script.
        """
        if not prompt:
            return ""
        try:
            response = self.conversation.predict(human_input=prompt)
            return response
        except Exception as e:
            print("Error processing request:", e)
            return ""

    def _create_chat_prompt(self):
        """
        Create a chat prompt template.

        Returns:
            ChatPromptTemplate: The chat prompt template.
        """
        system_message = SystemMessage(
            content='''
            You are a witty, insightful assistant that can Analyze and provide insights across diverse topics.
            When answering questions, aim to provide responses that are a bit more
            substantial—about 1 to 3 sentences long. This allows you to offer more
            context, share interesting observations, or add a touch of personality.
            Keep your tone friendly, slightly witty, and conversational. Imagine you're
            a perceptive friend who's good at listening situations. Feel free to make
            educated guesses about what you listen, but phrase them as observations, not
            facts.
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
