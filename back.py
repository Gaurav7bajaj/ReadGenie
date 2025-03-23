from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
import pytesseract
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import io
import requests
from config import TESSERACT_CMD, OLLAMA_URL, MODEL_NAME

app = FastAPI()

# Load models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS index
dimension = 384  # Output size of the embedding model
faiss_index = faiss.IndexFlatL2(dimension)
stored_texts = []

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

class TextRequest(BaseModel):
    text: str

class QuestionRequest(BaseModel):
    question: str

@app.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        extracted_text = pytesseract.image_to_string(image).strip()

        if not extracted_text:
            return {"error": "No text detected. Try using a clearer image."}

        return {"extracted_text": extracted_text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/summarize_text")
async def summarize_text(data: TextRequest):
    try:
        text = data.text.strip()
        if not text:
            return {"error": "No text provided"}

        max_input_length = 1024
        if len(text) > max_input_length:
            text = text[:max_input_length]

        summary = summarizer(text, max_length=500, min_length=50, do_sample=False)
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        return {"error": str(e)}

@app.post("/store_text")
async def store_text(data: TextRequest):
    global stored_texts, faiss_index

    text = data.text.strip()
    if not text:
        return {"error": "No text provided"}

    # Generate embeddings
    embeddings = embedding_model.encode([text])
    faiss_index.add(np.array(embeddings, dtype=np.float32))

    stored_texts.append(text)

    return {"message": "Text stored successfully"}

@app.post("/answer_question")
async def answer_question(data: QuestionRequest):
    global stored_texts, faiss_index

    question = data.question.strip()
    if not question:
        return {"error": "No question provided"}

    # Generate embedding for the question
    question_embedding = embedding_model.encode([question])
    _, indices = faiss_index.search(np.array(question_embedding, dtype=np.float32), 1)

    if indices[0][0] == -1 or len(stored_texts) == 0:
        return {"answer": "No relevant text found."}

    retrieved_text = stored_texts[indices[0][0]]

    # Send Query to Ollama (Gemma-3)
    payload = {
        "model": MODEL_NAME,
        "prompt": f"Context: {retrieved_text}\n\nQuestion: {question}\n\nAnswer:",
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    
    if response.status_code == 200:
        answer_data = response.json()
        return {"answer": answer_data.get("response", "No answer found.")}

    return {"error": "Error processing question with Ollama"}
