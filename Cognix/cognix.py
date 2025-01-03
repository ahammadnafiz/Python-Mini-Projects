import streamlit as st
import pypdf
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from streamlit_agraph import agraph, Node, Edge, Config
from typing import Dict, List
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
import json
import re
from neo4j import GraphDatabase

nltk.download('punkt')
nltk.download('punkt_tab')

st.set_page_config(
    page_title="Research Paper Analyzer",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

class ResearchPaperAnalyzer:
    def __init__(self):
        self.initialize_components()
        self.load_environment()
        self.setup_neo4j()
        self.setup_vector_store()
        self.setup_conversation_chain()

    def initialize_components(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.paper_text = ""
        self.paper_sections = {}
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def load_environment(self):
        load_dotenv()
        self.groq_model = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768"
        )

    def setup_neo4j(self):
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

    def setup_vector_store(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "cognix"
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric='cosine',
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
        self.vector_store = self.pc.Index(self.index_name)

    def setup_conversation_chain(self):
        self.conversation_memory = ConversationBufferWindowMemory(k=5)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a research paper analysis assistant. Provide detailed, accurate answers based on the paper's content."),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        self.conversation = ConversationChain(
            llm=self.groq_model,
            memory=self.conversation_memory,
            prompt=prompt
        )

    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            with st.spinner("Extracting text from PDF..."):
                pdf_reader = pypdf.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                self.paper_text = text
                self.extract_sections(text)
                return text
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")
            return ""

    def extract_sections(self, text: str):
        sections = {
            "Abstract": r"abstract",
            "Introduction": r"introduction",
            "Methods": r"methods|methodology",
            "Results": r"results",
            "Discussion": r"discussion",
            "Conclusion": r"conclusion",
            "References": r"references"
        }

        current_section = "Unknown"
        section_text = ""

        for line in text.split('\n'):
            line = line.strip()
            found_section = False

            for section, pattern in sections.items():
                if re.match(rf"^[0-9.\s]*{pattern}[s]?\s*$", line, re.IGNORECASE):
                    if section_text:
                        self.paper_sections[current_section] = section_text.strip()
                    current_section = section
                    section_text = ""
                    found_section = True
                    break

            if not found_section:
                section_text += line + "\n"

        if section_text:
            self.paper_sections[current_section] = section_text.strip()

    def create_knowledge_graph(self, text: str):
        try:
            with st.spinner("Creating knowledge graph..."):
                sentences = sent_tokenize(text)

                with self.neo4j_driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")  # Clear existing graph

                    for sent in sentences:
                        doc = self.nlp(sent)
                        entities = [(ent.text, ent.label_) for ent in doc.ents]

                        for i, (entity, label) in enumerate(entities):
                            session.run(
                                "MERGE (e:Entity {name: $name, type: $type})",
                                name=entity, type=label
                            )

                            if i > 0:
                                prev_entity = entities[i-1][0]
                                session.run("""
                                    MATCH (e1:Entity {name: $name1})
                                    MATCH (e2:Entity {name: $name2})
                                    MERGE (e1)-[:RELATED_TO]->(e2)
                                """, name1=prev_entity, name2=entity)

                self._store_embeddings(text)
        except Exception as e:
            st.error(f"Error creating knowledge graph: {str(e)}")

    def _store_embeddings(self, text: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

            chunks = [text[i:i+512] for i in range(0, len(text), 512)]

            for i, chunk in enumerate(chunks):
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    embeddings = model(**inputs).last_hidden_state.mean(dim=1)

                self.vector_store.upsert(
                    vectors=[(str(i), embeddings[0].tolist(), {"text": chunk})],
                    namespace="research_paper"
                )
        except Exception as e:
            st.error(f"Error storing embeddings: {str(e)}")

    def query_knowledge_graph(self, question: str) -> str:
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $keyword
                    MATCH (e)-[*1..2]-(related)
                    RETURN e.name, related.name
                    LIMIT 5
                """, keyword=question.lower())

                context_entities = set()
                for record in result:
                    context_entities.add(record["e.name"])
                    context_entities.add(record["related.name"])

            relevant_sections = []
            for entity in context_entities:
                for section, content in self.paper_sections.items():
                    if entity.lower() in content.lower():
                        relevant_sections.append(f"{section}: {content[:500]}...")

            context = "\n".join(relevant_sections)

            prompt = f"""Based on the following context from the research paper:
            {context}

            Question: {question}

            Provide a specific answer using only information from the paper."""

            response = self.conversation.predict(input=prompt)

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            st.error(f"Error querying knowledge graph: {str(e)}")
            return "Sorry, I couldn't process your question."

    def _get_embedding(self, text: str) -> List[float]:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings[0].tolist()

    def visualize_knowledge_graph(self):
        nodes = []
        edges = []

        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN n")
            for record in result:
                node = record["n"]
                nodes.append(Node(
                    id=node["name"],
                    label=f"{node['name']}\n({node['type']})",
                    size=25,
                    color="#00a0dc"
                ))

            result = session.run("MATCH (n1:Entity)-[r]->(n2:Entity) RETURN n1.name, n2.name")
            for record in result:
                edges.append(Edge(
                    source=record["n1.name"],
                    target=record["n2.name"],
                    type="CURVE_SMOOTH"
                ))

        config = Config(width=None, height=600, directed=True)
        return nodes, edges, config

def render_chat_messages():
    for message in st.session_state.chat_history:
        message_class = "user-message" if message["role"] == "user" else "assistant-message"
        st.markdown(f"""
            <div class="chat-message {message_class}">
                <p>{message["content"]}</p>
            </div>
        """, unsafe_allow_html=True)

def main():
    st.title("📚 Research Paper Analyzer")

    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ResearchPaperAnalyzer()
    if 'paper_uploaded' not in st.session_state:
        st.session_state.paper_uploaded = False
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "Upload Paper"

    with st.sidebar:
        views = ["Upload Paper", "Paper Sections", "Chat Interface", "Knowledge Graph"]
        st.session_state.current_view = st.radio("Select View", views)

        if st.session_state.current_view == "Upload Paper":
            st.header("📄 Upload Paper")
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

            if uploaded_file:
                if st.button("Process Paper"):
                    text = st.session_state.analyzer.extract_text_from_pdf(uploaded_file)
                    if text:
                        st.session_state.analyzer.create_knowledge_graph(text)
                        st.session_state.paper_uploaded = True
                        st.success("Paper processed successfully!")

    if not st.session_state.paper_uploaded and st.session_state.current_view != "Upload Paper":
        st.info("👈 Please upload a research paper PDF first!")
        return

    if st.session_state.current_view == "Paper Sections" and st.session_state.paper_uploaded:
        st.header("📑 Paper Sections")
        if st.session_state.analyzer.paper_sections:
            sections = list(st.session_state.analyzer.paper_sections.keys())
            selected_section = st.radio("Select a section to view:", sections)

            st.markdown("### " + selected_section)
            st.markdown('<div class="section-content">', unsafe_allow_html=True)
            st.write(st.session_state.analyzer.paper_sections[selected_section])
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No sections found in the paper.")

    elif st.session_state.current_view == "Chat Interface" and st.session_state.paper_uploaded:
        st.header("💭 Chat Interface")
        render_chat_messages()

        chat_input = st.chat_input("Ask a question about the paper:")
        if chat_input:
            response = st.session_state.analyzer.query_knowledge_graph(chat_input)
            st.session_state.chat_history.append({"role": "user", "content": chat_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    elif st.session_state.current_view == "Knowledge Graph" and st.session_state.paper_uploaded:
        st.header("🕸️ Knowledge Graph")
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        nodes, edges, config = st.session_state.analyzer.visualize_knowledge_graph()
        agraph(nodes=nodes, edges=edges, config=config)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
