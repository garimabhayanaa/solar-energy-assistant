import os
import streamlit as st
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMChain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from collections import deque
from langchain.llms.base import LLM
from typing import Any, List, Optional

# Load API Key from Environment Variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"

# Conversation History
conversation_history = deque(maxlen=5)  # Keep last 5 messages

# Custom OpenRouter LLM Class
class OpenRouterLLM(LLM):
    openrouter_api_key: str
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.openrouter_api_key}"}
        data = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise exception for non-200 status codes
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return "I apologize, but I'm having trouble connecting to the API right now. Please try again later."
        except (KeyError, IndexError) as e:
            st.error(f"Response parsing error: {str(e)}")
            return "I received an unexpected response format. Please try again."

# Load Vector Store
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        raise

# Custom Prompt Engineering
def set_custom_prompt():
    template = """You are an AI assistant specialized in solar energy.
    Use the following pieces of context to provide accurate and helpful responses.
    If you don't know the answer, just say that you don't know.

    Context: {context}
    Question: {question}
    Assistant: """
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def format_chat_history(history_deque):
    return "\n".join(list(history_deque))

# Maintain Context
def maintain_context(user_query):
    global conversation_history
    conversation_history.append(f"User: {user_query}")
    return "\n".join(conversation_history)

# Main App Function
def main():
    try:
        if not OPENROUTER_API_KEY:
            st.error("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")
            return

        vectorstore = get_vectorstore()
        
        # Initialize OpenRouter LLM
        llm = OpenRouterLLM(openrouter_api_key=OPENROUTER_API_KEY)
        
        # Create prompt
        prompt = set_custom_prompt()
        
        # Create LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        # Create StuffDocumentsChain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )
        
        # Create the QA chain
        qa = RetrievalQA(
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            combine_documents_chain=combine_documents_chain,
            return_source_documents=True
        )
        
        # Custom CSS for dark mode and white sidebar
        st.markdown("""
        <style>
        /* Main app background */
        .stApp {
            background-color: black;
            color: white;
            margin: 0 auto;
        }
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: white;
            color: black;
        }
        section[data-testid="stSidebar"] .stMarkdown {
            color: black;
        }
        section[data-testid="stSidebar"] button {
            background-color: #f0f2f6;
            color: black;
        }
        
        /* Chat message styling */
        [data-testid="stChatMessage"] {
            background-color: #1E2329;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        
        /* User message specific styling */
        [data-testid="stChatMessage"] [data-testid="StyledLinkIconContainer"] {
            background-color: #2E3339;
        }
        
        /* Chat input styling */
        .stTextInput input {
            background-color: #1E2329;
            color: white;
            border-color: #2E3339;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #2E3339;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.image("logo.png", width=100)
            st.title("☀️ Solar Industry Expert")
            
            # Add welcome message
            st.markdown("""
            <div style='background-color: #f0f2f6; color: black; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h3>Welcome to the Solar Energy Assistant! 👋</h3>
            
            This AI assistant can help you with:
            <ul>
            <li>Solar Panel Technology</li>
            <li>Installation</li>
            <li>Cost and Efficiency</li>
            <li>Maintenance</li>
            <li>Technical Specifications</li>
            <li>Industry Trends</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Clear Conversation", type="secondary"):
                st.session_state.messages = []
                conversation_history.clear()
                st.rerun()
        
        # Initialize chat interface
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            
            # Add welcome message with dark theme styling
            welcome_msg = {
                'role': 'assistant',
                'content': """Hi! I am your Solar Energy Assistant. How can I help you today?"""
            }
            st.session_state.messages.append(welcome_msg)
        
        # Display chat messages with dark theme styling
        for message in st.session_state.messages:
            with st.chat_message(message['role'], avatar="🧑" if message['role'] == "user" else "🤖"):
                st.write(message['content'])
        
        # Chat input
        if prompt := st.chat_input("💬 Ask about solar energy...", key="chat_input"):
            # User message
            with st.chat_message("user", avatar="🧑"):
                st.write(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            
            # Get response first
            try:
                with st.spinner("Thinking..."):
                    result = qa.invoke({"query": prompt})
                    response = result["result"]
                
                # Only create assistant message container after we have the response
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(response)
                    
                # Update conversation history
                conversation_history.append(f"User: {prompt}")
                conversation_history.append(f"Assistant: {response}")
                st.session_state.messages.append({'role': 'assistant', 'content': response})
            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
