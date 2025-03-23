import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("OCR RAG: Extract, Summarize & Answer Questions")

# File uploader for images
uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Extract & Summarize Text"):
        with st.spinner("Processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
                response = requests.post(f"{API_URL}/extract_text", files=files)

                if response.status_code == 200:
                    data = response.json()
                    extracted_text = data.get("extracted_text", "").strip()

                    if extracted_text:
                        st.success("Text Extracted Successfully!")
                        st.subheader("Extracted Text:")
                        st.text_area("", extracted_text, height=200)

                        # Summarization
                        summary_response = requests.post(f"{API_URL}/summarize_text", json={"text": extracted_text})

                        if summary_response.status_code == 200:
                            summary_data = summary_response.json()
                            summary = summary_data.get("summary", "").strip()
                            st.subheader("Summarized Text:")
                            st.text_area("", summary, height=100)

                        # Store text in FAISS for retrieval
                        store_response = requests.post(f"{API_URL}/store_text", json={"text": extracted_text})
                        if store_response.status_code == 200:
                            st.success("Text stored in FAISS successfully!")
                    else:
                        st.error("No text extracted. Try a clearer image.")

                else:
                    st.error("Error extracting text. Please try again.")

            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Make sure the FastAPI server is running.")

    # Question-Answering
    question = st.text_input("Ask a question about the text:")
    if st.button("Get Answer"):
        if question:
            try:
                qa_response = requests.post(f"{API_URL}/answer_question", json={"question": question})
                if qa_response.status_code == 200:
                    answer_data = qa_response.json()
                    st.subheader("Answer:")
                    st.write(answer_data.get("answer", "No answer found."))
                else:
                    st.error("Error in answering the question.")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Make sure the FastAPI server is running.")
