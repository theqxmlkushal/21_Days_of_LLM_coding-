#to run this app, make sure to install the required packages:
# pip install PyPDF2 faiss-cpu sentence-transformers tiktoken groq
import streamlit as st
import sys
from io import BytesIO
import os

import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import tiktoken
import re
from typing import List, Dict, Tuple
from groq import Groq

# Configuration parameters
CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k": 3,
    "llm_model": "llama-3.1-8b-instant",    # Groq model
    "max_tokens": 500,
    "temperature": 0.7
}

class ResearchPaperRAG:
    def __init__(self, pdf_path: str, config: Dict, groq_api_key: str):
        self.config = config
        self.groq_api_key = groq_api_key
        self.client = Groq(api_key=self.groq_api_key)

        self.pdf_text = self._load_pdf(pdf_path)
        self.chunks = self._chunk_text(self.pdf_text, self.config["chunk_size"], self.config["chunk_overlap"])
        self.embeddings, self.embedding_model = self._create_embeddings(self.chunks, self.config["embedding_model"])
        self.vector_index = self._build_faiss_index(self.embeddings)
        print("RAG system initialized successfully!")

    def _load_pdf(self, pdf_path: str) -> str:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def _create_embeddings(self, chunks: List[str], model_name: str):
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks)
        return embeddings, model

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def _retrieve_relevant_chunks(self, query: str, top_k: int) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            relevant_chunks.append({
                "id": idx,
                "text": self.chunks[idx],
                "distance": float(distances[0][i]),
                "similarity_score": 1 / (1 + float(distances[0][i])), # Convert distance to similarity score
                "length": len(self.chunks[idx])
            })
        return relevant_chunks

    def _generate_answer(self, query: str, relevant_chunks: List[Dict]) -> Dict:
        context = "\n\n".join([
            f"[Chunk {chunk['id'] + 1}]:\n{chunk['text']}"
            for chunk in relevant_chunks
        ])
        prompt = f"""You are a helpful research paper assistant. Your task is to answer questions based ONLY on the provided context from a research paper.\n\nContext from the paper: {context}\n\nQuestion: {query}\n\nInstructions:\n- Answer based ONLY on the information in the context above\n- If the context doesn't contain enough information to answer, say so clearly\n- Be concise but comprehensive\n- When possible, cite which chunk(s) you're using (e.g., "According to Chunk 1...")\n- Maintain an academic yet accessible tone\n\nAnswer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.config["llm_model"],
                messages=[
                    {"role": "system", "content": "You are a helpful research paper assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"]
            )
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "context_used": context,
                "chunks_used": len(relevant_chunks),
                "relevant_chunks": relevant_chunks,
                "model": self.config["llm_model"],
                "success": True
            }
        except Exception as e:
            print(f"❌ Error calling Groq LLM: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "context_used": context,
                "chunks_used": len(relevant_chunks),
                "relevant_chunks": relevant_chunks,
                "model": self.config["llm_model"],
                "success": False
            }

    def ask(self, query: str) -> Dict:
        relevant_chunks = self._retrieve_relevant_chunks(query, self.config["top_k"])
        return self._generate_answer(query, relevant_chunks)

    def get_stats(self) -> Dict:
        return {
            "total_chunks": len(self.chunks),
            "total_characters": len(self.pdf_text),
            "embedding_model": self.config["embedding_model"],
            "embedding_dimension": self.embeddings.shape[1]
        }




# Page configuration
st.set_page_config(
    page_title="Research Paper Explainer Bot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .chunk-box {
        background-color: #fff9e6;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ffd700;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Model settings
    st.markdown("### Embedding Model")
    embedding_model = st.selectbox(
        "Choose embedding model:",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        help="MiniLM is faster, mpnet is more accurate",
        index=["all-MiniLM-L6-v2", "all-mpnet-base-v2"].index(CONFIG["embedding_model"])
    )

    st.markdown("### Retrieval Settings")
    top_k = st.slider("Number of chunks to retrieve:", 1, 10, CONFIG["top_k"])
    chunk_size = st.slider("Chunk size (characters):", 200, 1000, CONFIG["chunk_size"], 50)
    chunk_overlap = st.slider("Chunk overlap:", 0, 200, CONFIG["chunk_overlap"], 10)

    st.markdown("### Generation Settings")
    llm_model = st.selectbox(
        "Groq Model:", # Changed to Groq Model
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"], # Updated with Groq models
        help="Choose a Groq LLM model",
        index=["llama-3.1-8b-instant", "llama-3.1-70b-versatile"].index(CONFIG["llm_model"]) if CONFIG["llm_model"] in ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"] else 0
    )
    temperature = st.slider("Temperature:", 0.0, 1.0, CONFIG["temperature"], 0.1)
    max_tokens = st.slider("Max tokens:", 100, 2000, CONFIG["max_tokens"], 50)

    # Update config (creates a new config object for the RAG system)
    current_config = {
        "embedding_model": embedding_model,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "llm_model": llm_model,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    st.markdown("---కు")

    # API Key input
    st.markdown("### 🔑 Groq API Key") # Changed to Groq API Key
    api_key = st.text_input("Enter your Groq API key:", type="password", key="groq_key") # Changed to Groq API key
    if api_key:
        st.success("✅ Groq API key set!")
        # Store in session state
        st.session_state.groq_api_key = api_key # Changed to groq_api_key
    else:
        st.warning("⚠️ Please enter your Groq API key to use the system")

    st.markdown("---కు")

    # Stats
    if st.session_state.rag_system:
        st.markdown("### 📊 System Stats")
        stats = st.session_state.rag_system.get_stats()
        st.metric("Total Chunks", stats["total_chunks"])
        st.metric("Characters", f"{stats['total_characters']:,}")
        st.metric("Embedding Dim", stats["embedding_dimension"])

# Main content
st.markdown('<h1 class="main-header">🔬 Research Paper Explainer Bot (Groq-Powered)</h1>', unsafe_allow_html=True) # Changed to Groq-Powered

st.markdown("""
Welcome! Upload a research paper (PDF) and ask questions about it.
The system uses RAG (Retrieval-Augmented Generation) with **Groq AI** to provide accurate answers based on the paper's content. # Changed to Groq AI
""")

# File upload
uploaded_file = st.file_uploader(
    "📄 Upload Research Paper (PDF)",
    type="pdf",
    help="Upload a research paper to analyze"
)

if uploaded_file is not None:
    # Check if this is a new file
    if st.session_state.current_pdf != uploaded_file.name:
        st.session_state.current_pdf = uploaded_file.name
        st.session_state.rag_system = None
        st.session_state.conversation_history = []

    # Save uploaded file temporarily
    with open("temp_paper.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Initialize RAG system if not already done
    if st.session_state.rag_system is None:
        # Check if API key is provided
        if 'groq_api_key' not in st.session_state or not st.session_state.groq_api_key: # Changed to groq_api_key
            st.error("❌ Please enter your Groq API key in the sidebar first!") # Changed to Groq API key
            st.stop()

        with st.spinner("🔄 Processing paper... This may take a minute..."):
            try:
                st.session_state.rag_system = ResearchPaperRAG(
                    "temp_paper.pdf",
                    current_config, # Pass the dynamically updated config
                    groq_api_key=st.session_state.groq_api_key # Changed to groq_api_key
                )
                st.success("✅ Paper processed successfully with Groq!") # Changed to Groq!

                # Show stats
                stats = st.session_state.rag_system.get_stats()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Chunks", stats["total_chunks"])
                col2.metric("Characters", f"{stats['total_characters']:,}")
                col3.metric("Model", stats["embedding_model"].split('-')[1])
                col4.metric("Dimension", stats["embedding_dimension"])

            except Exception as e:
                st.error(f"❌ Error processing paper: {str(e)}")
                st.stop()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "🔍 Retrieval Analysis", "📜 Conversation History"])

    with tab1:
        st.markdown("## Ask a Question")

        # Sample questions
        with st.expander("📋 Sample Questions"):
            sample_questions = [
                "What is the main contribution of this paper?",
                "What methodology did the authors use?",
                "What are the key results or findings?",
                "What datasets were used?",
                "What are the limitations?",
                "What future work is suggested?"
            ]
            for sq in sample_questions:
                if st.button(sq, key=sq):
                    st.session_state.current_question = sq

        # Question input
        question = st.text_area(
            "Your question:",
            value=st.session_state.get('current_question', ''),
            height=100,
            placeholder="e.g., What is the main contribution of this paper?"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)
        with col2:
            show_context = st.checkbox("Show retrieved context", value=True)

        if ask_button and question:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.rag_system.ask(question)

                    # Display answer
                    st.markdown("### 🤖 Answer")
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>',
                              unsafe_allow_html=True)

                    # Save to history
                    st.session_state.conversation_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "chunks": result["relevant_chunks"]
                    })

                    # Show metadata
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Chunks Used", result["chunks_used"])
                    col2.metric("Model", result["model"])
                    col3.metric(
                        "Top Similarity",
                        f"{result['relevant_chunks'][0]['similarity_score']:.3f}"
                    )

                    # Show context if requested
                    if show_context:
                        st.markdown("### 📚 Retrieved Context")
                        for i, chunk in enumerate(result["relevant_chunks"], 1):
                            with st.expander(
                                f"Chunk {chunk['id'] + 1} (Score: {chunk['similarity_score']:.3f})" # Changed ID to id + 1
                            ):
                                st.markdown(f'<div class="chunk-box">{chunk["text"]}</div>', # Changed ID to id + 1
                                          unsafe_allow_html=True)
                                st.caption(f"Characters: {chunk['length']}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with tab2:
        st.markdown("## 🔍 Retrieval Analysis")
        st.markdown("Analyze how well the system retrieves relevant context for your questions.")

        analysis_query = st.text_input(
            "Enter a query to analyze:",
            placeholder="What is the main contribution?"
        )

        if st.button("Analyze Retrieval") and analysis_query:
            st.markdown("### Retrieved Chunks")

            relevant = st.session_state.rag_system.ask(analysis_query)["relevant_chunks"]

            for i, chunk in enumerate(relevant, 1):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**Chunk {chunk['id'] + 1} (Rank {i})**") # Changed ID to id + 1
                    st.text(chunk['text'][:300] + "...")

                with col2:
                    st.metric("Similarity", f"{chunk['similarity_score']:.3f}")
                    st.metric("Length", f"{chunk['length']} chars")

                st.markdown("---కు")

            # Show statistics
            scores = [c['similarity_score'] for c in relevant]
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Score", f"{sum(scores)/len(scores):.3f}")
            col2.metric("Min Score", f"{min(scores):.3f}")
            col3.metric("Max Score", f"{max(scores):.3f}")

    with tab3:
        st.markdown("## 📜 Conversation History")

        if st.session_state.conversation_history:
            st.markdown(f"**Total Questions Asked:** {len(st.session_state.conversation_history)}")

            # Export button
            if st.button("💾 Export Conversation"):
                conversation_text = ""
                for i, conv in enumerate(st.session_state.conversation_history, 1):
                    conversation_text += f"\n{'='*60}\n"
                    conversation_text += f"Q{i}: {conv['question']}\n"
                    conversation_text += f"{'='*60}\n"
                    conversation_text += f"A: {conv['answer']}\n\n"

                st.download_button(
                    "📥 Download as TXT",
                    conversation_text,
                    file_name="conversation_history.txt",
                    mime="text/plain"
                )

            st.markdown("---కు")

            # Display history
            for i, conv in enumerate(reversed(st.session_state.conversation_history), 1):
                with st.expander(f"Q{len(st.session_state.conversation_history)-i+1}: {conv['question'][:100]}..."):
                    st.markdown("**Question:**")
                    st.markdown(f'<div class="question-box">{conv["question"]}</div>',
                              unsafe_allow_html=True)

                    st.markdown("**Answer:**")
                    st.markdown(f'<div class="answer-box">{conv["answer"]}</div>',
                              unsafe_allow_html=True)

                    st.caption(f"Used {len(conv['chunks'])} chunks")
        else:
            st.info("No questions asked yet. Go to the 'Ask Questions' tab to start!")

else:
    # Instructions when no file is uploaded
    st.info("👆 Upload a research paper PDF to get started!")

    st.markdown("### How it works:")
    st.markdown("""
    1. **Upload** your research paper (PDF format)
    2. **Wait** for the system to process it (chunks, embeddings, indexing)
    3. **Ask** questions about the paper
    4. **Get** answers powered by Groq AI to provide accurate answers based on the paper's content with relevant citations # Changed to Groq AI
    """)

    st.markdown("### Why Groq?") # Changed to Groq
    st.markdown("""
    - ✅ **Large context window** (131k tokens vs 4k-8k for others)
    - ✅ **Fast inference** speed
    - ✅ **High quality** responses
    - ✅ **OpenAI-compatible** API
    """)

    st.markdown("### Tips:")
    st.markdown("""
    - ✅ Ask specific questions for better answers
    - ✅ Check the retrieved context to see what information was used
    - ✅ Adjust settings in the sidebar to tune performance
    - ✅ Use the sample questions as templates
    """)

# Footer
st.markdown("---కు")
st.markdown("""
<div style='text-align: center; color: gray;'>
Built with ❤️ using Streamlit + Groq AI | RAG Pipeline without frameworks # Changed to Groq AI
</div>
""", unsafe_allow_html=True)
