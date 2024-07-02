import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up Streamlit UI
st.set_page_config(page_title="InnoGenie", page_icon="💡")
st.image('Assets/InnoGenie.png')
st.caption("Powered by LLaMA3 70b, Langchain, and Groq API")

# Sidebar for API key input
with st.sidebar:
    st.subheader("API Key Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    st.markdown("[Get a GROQ API key](https://console.groq.com/keys)")

# Function to handle API key validation
def handle_api():
    if not groq_api_key:
        st.warning("Please enter your GROQ API key.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key

# Call function to handle API key input and validation
handle_api()

# Validate and initialize ChatGroq instance
try:
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama3-70b-8192",
        max_tokens=8192
    )
except ValueError as ve:
    st.error(f"Error initializing ChatGroq: {ve}")
    st.stop()

# Define the structure for an idea
class Idea(BaseModel):
    title: str = Field(description="The title of the idea")
    description: str = Field(description="A brief description of the idea")
    potential_challenges: List[str] = Field(description="List of potential challenges or considerations")
    next_steps: List[str] = Field(description="List of next steps to develop this idea")
    market_potential: str = Field(description="Brief analysis of market potential")
    innovation_factor: str = Field(description="What makes this idea innovative")

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
   - You don't make unfounded claims or speculate beyond reasonable extrapolation
   - You consider ethical implications and societal impact
   - Ideas are distinct from each other and not merely variations of the same concept

4. If you're unsure about any aspect, state it clearly instead of making assumptions.

5. Use the following format for your output:
{format_instructions}

Remember, quality and practicality are more important than complexity or futuristic appeal. Focus on ideas that could be realistically pursued by an entrepreneur or organization.
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

# User input form
with st.form("idea_input"):
    area = st.text_input("Area of Interest")
    tags_input = st.text_input("Tags (comma-separated)")
    subcategory = st.text_input("Subcategory")
    submit_button = st.form_submit_button("Generate Ideas")

# Generate ideas
if submit_button:
    if not area or not tags_input or not subcategory:
        st.error("Please fill in all fields before generating ideas.")
    else:
        # Process tags
        tags = ", ".join([tag.strip() for tag in tags_input.split(',') if tag.strip()])
        
        with st.spinner("Generating ideas..."):
            try:
                st.write(f"Debug - Area: {area}, Tags: {tags}, Subcategory: {subcategory}")
                ideas_output = idea_chain.invoke({
                    "area": area,
                    "tags": tags,
                    "subcategory": subcategory,
                    "format_instructions": parser.get_format_instructions()
                })
                # st.write(f"Debug - LLMChain output: {ideas_output}")
                ideas_list = parser.parse(ideas_output['text'])
                st.session_state.generated_ideas = ideas_list.ideas
            except ValueError as ve:
                st.error(f"Value Error: {str(ve)}")
                st.write(f"Debug - Value Error details: {ve}")
            except KeyError as ke:
                st.error(f"Key Error: {str(ke)}")
                st.write(f"Debug - Key Error details: {ke}")
            except Exception as e:
                st.error(f"An error occurred while generating ideas: {str(e)}")
                st.write(f"Debug - Error details: {type(e).__name__}, {str(e)}")

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

# Idea exploration
if 'generated_ideas' in st.session_state:
    st.subheader("Explore an Idea")
    selected_idea_index = st.selectbox("Select an idea to explore:", 
                                       range(len(st.session_state.generated_ideas)), 
                                       format_func=lambda i: st.session_state.generated_ideas[i].title)
    selected_idea = st.session_state.generated_ideas[selected_idea_index]
    question = st.text_input("Ask a question about this idea:")
    if st.button("Explore") and question:
        with st.spinner("Exploring the idea..."):
            try:
                exploration = exploration_chain.invoke({
                    "idea": json.dumps(selected_idea.dict()),
                    "question": question
                })
                st.write(exploration['text'])
            except Exception as e:
                st.error(f"An error occurred while exploring the idea: {str(e)}")

# Debugging information
if st.checkbox("Show Debug Info"):
    st.subheader("Debug Info")
    st.write("Session State:")
    st.write(st.session_state)
