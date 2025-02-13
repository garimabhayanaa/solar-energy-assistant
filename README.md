# Solar Energy Assistant 🌞

A sophisticated AI-powered chatbot specializing in solar energy knowledge, built with Streamlit, LangChain, and OpenRouter API.

## Features ✨

- Interactive chat interface with a dark theme
- Context-aware responses using FAISS vector store
- PDF document processing for knowledge base
- Conversation history management
- Responsive UI with custom styling
- Error handling and graceful degradation

## Tech Stack 🛠️

- **Frontend**: Streamlit
- **Language Model**: Mistral 7B (via OpenRouter API)
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Document Processing**: LangChain
- **PDF Processing**: PyPDF Loader

## Installation 🚀

1. Clone the repository:
bash
git clone <your-repository-url>
cd solar-energy-assistant

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export OPENROUTER_API_KEY=your_api_key_here
```

5. Create a `data` directory and add your PDF documents:
```bash
mkdir data
# Add your PDF files to the data directory
```

## Usage 💡

1. First, process the PDF documents to create the vector store:
```bash
python create_memory_for_llm.py
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Project Structure 📁

```
solar-energy-assistant/
├── app.py                    # Main Streamlit application
├── create_memory_for_llm.py  # PDF processing and vector store creation
├── data/                     # Directory for PDF documents
├── vectorstore/             # FAISS vector store
├── logo.png                 # Application logo
└── requirements.txt         # Project dependencies
```

## Features in Detail 🔍

- **Context-Aware Responses**: Utilizes FAISS vector store to provide relevant information from the knowledge base
- **Conversation History**: Maintains context of the last 5 messages
- **Custom Styling**: Dark theme with professional UI elements
- **Error Handling**: Robust error handling for API calls and document processing
- **Clear Conversation**: Option to reset the chat history
- **Loading States**: Visual feedback during API calls
- **Sidebar Information**: Quick access to capabilities and features

// ... existing code ...

## Implementation Details 🔧

### Vector Store Creation
The application processes PDF documents in two main steps:

1. **Document Processing**:
   - PDFs are loaded using `PyPDFLoader`
   - Documents are split into chunks of 500 characters with 50-character overlap
   - Text chunks are processed using RecursiveCharacterTextSplitter

2. **Embedding Creation**:
   - Text chunks are converted to embeddings using HuggingFace's sentence-transformers
   - Embeddings are stored in a FAISS vector store for efficient similarity search

### Chat Implementation

1. **Question-Answering Chain**:
   ```python
   qa = RetrievalQA.from_chain_type(
       llm=llm,
       chain_type="stuff",
       retriever=vectorstore.as_retriever(),
       return_source_documents=True
   )
   ```

2. **Memory Management**:
   - Maintains conversation history of last 5 messages
   - Uses session state to persist chat across reruns
   - Implements clear conversation functionality

3. **LLM Integration**:
   - Uses OpenRouter API to access Mistral 7B
   - Custom LLM class implementation for API communication
   - Handles rate limiting and API errors

### Security Measures

1. **API Security**:
   - API keys stored in environment variables
   - No hardcoded credentials
   - Secure API communication with error handling

2. **Data Safety**:
   - Local vector store implementation
   - Safe deserialization of FAISS index
   - Input validation for user queries

### Performance Optimizations

1. **Vector Search**:
   - FAISS for efficient similarity search
   - Optimized chunk size for better context retrieval
   - Minimal chunk overlap to reduce redundancy

2. **UI Performance**:
   - Lazy loading of components
   - Efficient state management
   - Optimized rerun triggers

### Custom Components

1. **Styling**:
   ```python
   st.markdown("""
       <style>
           .stApp {
               background-color: #0E1117;
               color: white;
           }
       </style>
   """, unsafe_allow_html=True)
   ```

2. **Chat Interface**:
   - Custom avatars for user and assistant
   - Markdown support for formatted responses
   - Loading indicators for API calls

### Error Handling Strategy

1. **Graceful Degradation**:
   - Comprehensive try-except blocks
   - User-friendly error messages
   - Fallback responses for API failures

2. **Input Validation**:
   - PDF document validation
   - API key verification
   - User input sanitization

## Error Handling 🛡️

The application includes comprehensive error handling for:
- PDF document loading
- Vector store creation and loading
- API communication
- User input processing
- File system operations

## Sample Conversation 💬
### Q: How do monocrystalline and polycrystalline panels compare in efficiency?
### A: Monocrystalline solar panels generally have a higher efficiency compared to polycrystalline solar panels. This means that monocrystalline panels can convert a higher percentage of sunlight into electricity. However, the exact efficiency can vary between manufacturers and models, so it's always a good idea to check the specific efficiency rating when comparing panels.
### Q: How often should I clean my solar panels?
### A: It is recommended to clean your solar panels as needed, especially in dusty areas. A general guideline is to clean them every 6 months, but this can vary depending on the location and environmental conditions. If you live in an area with high dust levels or frequent storms, you might need to clean them more frequently. Always use a soft brush or a soft cloth and a non-abrasive, PH-neutral cleaning solution or a specialty solar panel cleaner. Avoid using high-pressure water as it can damage the solar cells.
### Q: What are the benefits of monocrystalline solar panels?
### A: Monocrystalline solar panels have several benefits, including higher efficiency compared to similarly rated polycrystalline panels, which means they can produce more electricity from a smaller space. This makes them a good choice for homeowners with limited roof area. They are also slightly more durable due to their high-purity silicon construction. The uniform look and even coloring of monocrystalline solar cells can also add an aesthetic appeal. However, the higher efficiency of monocrystalline panels means they tend to be slightly more expensive than polycrystalline panels.

## Future Improvements 🚀
- **Fine-Tuning Responses:** Experiment with different OpenRouter models.
- **Enhanced Context Memory:** Improve multi-turn conversation handling.
- **User Feedback Mechanism:** Allow users to rate responses.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments 🙏

- OpenRouter API for providing access to Mistral 7B
- Streamlit for the wonderful web framework
- LangChain for the powerful LLM tools
