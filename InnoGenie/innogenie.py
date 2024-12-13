import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List
import json
from dotenv import load_dotenv
import os
import random
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up Streamlit UI
st.set_page_config(page_title="InnoGenie 2.0", page_icon="💡")
st.image('Assets/InnoGenie.png')
st.caption("Powered by LLaMA3 70b, Langchain, and Groq API")

# Sidebar for API key input and creativity level
with st.sidebar:
    st.subheader("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    st.markdown("[Get a GROQ API key](https://console.groq.com/keys)")
    
    st.subheader("Customize Your Experience")
    creativity_level = st.slider("Creativity Level", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

def validate_groq_api_key(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json={"model": "llama3-70b-8192", "messages": [{"role": "user", "content": "Test"}]}
    )
    return response.status_code == 200

# In the sidebar:
if groq_api_key:
    if validate_groq_api_key(groq_api_key):
        st.sidebar.success("API Key validated successfully!")
    else:
        st.sidebar.error("Invalid API Key. Please check and try again.")
        st.stop()

# Function to handle API key validation
def handle_api():
    if not groq_api_key:
        st.warning("Please enter your GROQ API key.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key

# Call function to handle API key input and validation
handle_api()

# Validate and initialize ChatGroq instance
@st.cache_resource
def initialize_llm(temperature):
    try:
        return ChatGroq(
            temperature=temperature,
            model_name="llama3-70b-8192",
            max_tokens=8192
        )
    except Exception as e:
        logger.error(f"Error initializing ChatGroq: {e}")
        st.error(f"Error initializing ChatGroq: {e}")
        st.stop()

llm = initialize_llm(creativity_level)

# Define the structure for an idea with improved validation
class Idea(BaseModel):
    title: str = Field(description="The title of the idea")
    description: str = Field(description="A brief description of the idea")
    potential_challenges: List[str] = Field(description="List of potential challenges or considerations")
    next_steps: List[str] = Field(description="List of next steps to develop this idea")
    market_potential: str = Field(description="Brief analysis of market potential")
    innovation_factor: str = Field(description="What makes this idea innovative")

    @field_validator('title')
    def title_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v

    @field_validator('description')
    def description_length(cls, v):
        if len(v.split()) < 10:
            raise ValueError('Description should be at least 10 words')
        return v

class IdeaList(BaseModel):
    ideas: List[Idea] = Field(description="List of generated ideas")

# Create a parser
parser = PydanticOutputParser(pydantic_object=IdeaList)

# Enhanced prompt templates
idea_generation_template = PromptTemplate(
    input_variables=["area", "tags", "subcategory", "format_instructions"],
    template="""You are an expert idea generator with deep knowledge across various industries and technologies. Your task is to generate 5 innovative, practical, and implementable ideas for the given area, tags, and subcategory.

Area of Interest: {area}
Tags: {tags}
Subcategory: {subcategory}

Instructions:
1. Generate 5 distinct ideas that are:
   - Innovative: Offer a new approach or solution
   - Practical: Implementable with current or near-future technology
   - Relevant: Directly related to the given area, tags, and subcategory
   - Diverse: Cover different aspects or approaches within the given parameters

2. For each idea, provide:
   - A concise, catchy title (max 10 words)
   - A brief description (2-3 sentences) explaining the core concept
   - 2-3 potential challenges or considerations for implementation
   - 2-3 concrete next steps for developing the idea
   - A brief analysis of market potential (1-2 sentences)
   - What makes this idea innovative (1 sentence)

3. Ensure that:
   - Ideas are grounded in reality and current technological capabilities
   - You consider ethical implications and societal impact
   - Ideas are distinct from each other and not merely variations of the same concept

4. Use the following format for your output:
{format_instructions}

Remember, quality and practicality are key. Focus on ideas that could be realistically pursued by an entrepreneur or organization.
"""
)

idea_exploration_template = PromptTemplate(
    input_variables=["idea", "question"],
    template="""You are an expert consultant with deep knowledge in various fields of innovation and business development. Your task is to provide a detailed, informative, and practical response to a question about a specific idea.

Idea Details:
{idea}

Question: {question}

Instructions for providing your response:
1. Carefully analyze the idea and the specific question asked.
2. Provide a comprehensive answer that directly addresses the question.
3. Base your response on factual information, current market trends, and established business principles.
4. If the question touches on technical aspects, provide explanations that are accurate and up-to-date with current technology.
5. Include practical advice or actionable insights where relevant.
6. If any part of your response involves estimation or projection, clearly state this and provide the reasoning behind your estimates.
7. If the question asks for an opinion, clearly frame it as such and provide the rationale behind your viewpoint.
8. If you're unsure about any aspect, explicitly state that it would require further research or expert consultation in that specific area.
9. Consider potential ethical implications or societal impacts in your response where relevant.
10. Avoid making speculative claims or providing information that you're not confident about.
11. If the question is outside the scope of the provided idea or your expertise, state this clearly and suggest where the questioner might find more appropriate information.

Your response should be well-structured, using paragraphs or bullet points as appropriate for clarity. Aim to provide valuable, actionable insights that could genuinely assist in developing or implementing the idea.
"""
)

# Create LLMChains
idea_chain = LLMChain(llm=llm, prompt=idea_generation_template)
exploration_chain = LLMChain(llm=llm, prompt=idea_exploration_template)

# Define a new prompt template for generating inspiration
inspiration_template = PromptTemplate(
    input_variables=[],
    template="""
You are an innovative AI designed to spark creativity and generate unique, actionable inspiration prompts for idea generation. Your task is to provide one highly creative, thought-provoking inspiration prompt for generating innovative ideas. Consider emerging trends, interdisciplinary approaches, and unconventional thinking.

Instructions:
1. Craft a prompt that encourages thinking outside the box and exploring new possibilities.
2. Ensure the prompt is relevant to current or near-future technologies and societal needs.
3. Keep the prompt concise yet impactful, providing enough detail to stimulate creative thinking.

Provide the inspiration prompt below:
"""
)

# Create an LLMChain for generating inspiration
inspiration_chain = LLMChain(llm=llm, prompt=inspiration_template)

# Updated method to get random inspiration from the AI model
@st.cache_data(ttl=3600)
def get_random_inspiration():
    try:
        inspiration_output = inspiration_chain.invoke({})
        return inspiration_output['text'].strip()
    except Exception as e:
        logger.error(f"An error occurred while generating inspiration: {e}")
        # Fallback to a predefined list in case of an error
        inspirations = [
            "Think about combining two unrelated industries.",
            "How could emerging technologies solve this problem?",
            "Consider the needs of an underserved market segment.",
            "Imagine a world where resources are unlimited.",
            "What if you could break one fundamental law of physics?",
        ]
        return random.choice(inspirations)

# User input form with visual enhancements
st.subheader("Generate Innovative Ideas")
with st.form("idea_input"):
    col1, col2 = st.columns(2)
    with col1:
        area = st.text_input("Area of Interest", help="E.g., Healthcare, Education, Transportation")
        tags_input = st.text_input("Tags (comma-separated)", help="E.g., AI, sustainability, mobile")
    with col2:
        subcategory = st.text_input("Subcategory", help="E.g., Preventive care, Online learning, Last-mile delivery")
    submit_button = st.form_submit_button("Generate Ideas", use_container_width=True)

if st.button("Get Inspiration"):
    with st.expander('Inspiration'):
        st.info(get_random_inspiration())

# Generate ideas
if submit_button:
    if not area or not tags_input or not subcategory:
        st.error("Please fill in all fields before generating ideas.")
    else:
        # Process tags
        tags = ", ".join([tag.strip() for tag in tags_input.split(',') if tag.strip()])
        
        with st.spinner("Generating ideas..."):
            try:
                ideas_output = idea_chain.invoke({
                    "area": area,
                    "tags": tags,
                    "subcategory": subcategory,
                    "format_instructions": parser.get_format_instructions()
                })
                ideas_list = parser.parse(ideas_output['text'])
                st.session_state.generated_ideas = ideas_list.ideas
            except ValueError as ve:
                logger.error(f"Value Error: {ve}")
                st.error(f"An error occurred while processing the generated ideas. Please try again.")
            except KeyError as ke:
                logger.error(f"Key Error: {ke}")
                st.error("An unexpected error occurred. Please check your inputs and try again.")
            except Exception as e:
                logger.error(f"An error occurred while generating ideas: {e}")
                st.error("An unexpected error occurred. Please try again later.")

# Display generated ideas
if 'generated_ideas' in st.session_state:
    st.subheader("Generated Ideas")
    for i, idea in enumerate(st.session_state.generated_ideas, 1):
        with st.expander(f"Idea {i}: {idea.title}"):
            st.write(f"Description: {idea.description}")
            st.write("Potential Challenges:")
            for challenge in idea.potential_challenges:
                st.write(f"- {challenge}")
            st.write("Next Steps:")
            for step in idea.next_steps:
                st.write(f"- {step}")
            st.write(f"Market Potential: {idea.market_potential}")
            st.write(f"Innovation Factor: {idea.innovation_factor}")

if 'generated_ideas' in st.session_state:
    st.subheader("Explore an Idea")
    selected_idea_index = st.selectbox("Select an idea to explore:", 
                                       range(len(st.session_state.generated_ideas)), 
                                       format_func=lambda i: st.session_state.generated_ideas[i].title)
    selected_idea = st.session_state.generated_ideas[selected_idea_index]
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        st.chat_message(message['role']).write(message['content'])

    chat_input = st.chat_input("Ask a question about this idea:", key="chat_input")

    if chat_input:
        st.session_state.chat_history.append({'role': 'user', 'content': chat_input})
        with st.spinner("Exploring the idea..."):
            try:
                exploration = exploration_chain.invoke({
                    "idea": json.dumps(selected_idea.dict()),
                    "question": chat_input
                })
                st.session_state.chat_history.append({'role': 'assistant', 'content': exploration['text']})
                st.rerun()
            except Exception as e:
                logger.error(f"An error occurred while exploring the idea: {e}")
                st.error("An error occurred while exploring the idea. Please try again.")

# Save and load ideas
if 'generated_ideas' in st.session_state:
    if st.button("Save Ideas"):
        st.session_state.saved_ideas = st.session_state.generated_ideas
        st.success("Ideas saved successfully!")

if 'saved_ideas' in st.session_state:
    if st.button("Load Saved Ideas"):
        st.session_state.generated_ideas = st.session_state.saved_ideas
        st.success("Saved ideas loaded successfully!")

# Debugging information (only shown when "Show Debug Info" is checked)
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.subheader("Debug Info")
    st.sidebar.write("Session State:")
    st.sidebar.write(st.session_state)