from chat_interface import display_chat_interface
import streamlit as st
from api_utils import upload_document, list_documents, delete_document


st.title("Conversational RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "model" not in st.session_state:
    st.session_state.model = "llama-3.3-70b-versatile"

if "documents" not in st.session_state:
    st.session_state.documents = list_documents()


def display_sidebar():

    model_options = ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"]
    st.sidebar.selectbox("Select Model", options=model_options, key="model")

    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "docx", "html"], accept_multiple_files=True)
    if uploaded_file is not None:
        if st.sidebar.button("Upload"):
            with st.spinner("Uploading..."):
                upload_response = upload_document(uploaded_file)
                if upload_response:
                    st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully with ID {upload_response['file_id']}.")
                    st.session_state.documents = list_documents() # Refresh


    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        with st.spinner("Refreshing"):
            st.session_state.documents = list_documents()
    
    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()
    
    documents = st.session_state.documents
    if documents:
        for doc in documents:
            st.sidebar.text(f"{doc['filename']} (ID: {doc['id']}, Uploaded: {doc['upload_timestamp']})")
        
        selected_file_id = st.sidebar.selectbox("Select a document to delete", options=[doc['id'] for doc in documents], format_func=lambda x: next(doc['filename'] for doc in documents if doc['id'] == x))
        if st.sidebar.button("Deleted Selected Document"):
            with st.spinner("Deleting..."):
                delete_response = delete_document(selected_file_id)
                if delete_response:
                    st.sidebar.success(f"Document with ID {selected_file_id} deleted successfully.")
                    st.session_state.documents = list_documents()
                else:
                    st.sidebar.error(f"failed to delete document with ID {selected_file_id}.")


display_sidebar()
display_chat_interface()