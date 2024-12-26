import streamlit as st
from backend_app import index_research_paper, chatbot_query

# Initialize session state for indexing
if "is_indexed" not in st.session_state:
    st.session_state.is_indexed = False  # Initially, the paper is not indexed

st.title("Research Paper Chatbot")
st.write("Upload a research paper and ask questions about it!")

# File upload widget
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and not st.session_state.is_indexed:
    # Save the uploaded file locally for processing
    pdf_path = f"./uploaded_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Index the research paper
    with st.spinner("Indexing the research paper..."):
        index_research_paper(pdf_path)
    st.session_state.is_indexed = True  # Mark as indexed
    st.success("Research paper indexed successfully!")  # Notify the user

# User query input field
if st.session_state.is_indexed:
    user_question = st.text_input("Ask a question about the research paper:")

    if user_question:
        with st.spinner("Generating the answer..."):
            try:
                answer = chatbot_query(user_question)
                st.write(f"**Answer:** {answer}")  # Display the chatbot's response
            except Exception as e:
                st.error(f"An error occurred: {e}")  # Display an error message
else:
    st.warning("Please upload and index a research paper before asking questions.")
