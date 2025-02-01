from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, InputExample, losses
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import os
import pickle
from torch.utils.data import DataLoader
import datetime
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Download NLTK data
nltk.download('punkt')

CHAT_HISTORY_DIR = "chat-history"

if not os.path.exists(CHAT_HISTORY_DIR):
    os.makedirs(CHAT_HISTORY_DIR)

# Step 1: Extract text from PDFs
def extract_text_from_pdf(pdf_paths):
    text = ""
    for path in pdf_paths:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

# Step 2: Preprocess text into smaller chunks
def preprocess_text(text, chunk_size=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Step 3: Encode document chunks using a pre-trained model
def create_index(chunks, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=True)
    embeddings = embeddings.cpu().numpy()

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, model, chunks

# Step 4: Save and Load FAISS Index
def save_faiss_index(index, chunks, file_path="index.faiss", metadata_path="chunks.pkl"):
    faiss.write_index(index, file_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

def load_faiss_index(file_path="index.faiss", metadata_path="chunks.pkl"):
    if not os.path.exists(file_path) or not os.path.exists(metadata_path):
        return None, None  # Return None if index is missing
    index = faiss.read_index(file_path)
    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Step 5: Query the chatbot
def query_chatbot(question, index, model, chunks, summarizer, top_k=5, similarity_threshold=0.5):
    greetings = ["hi", "hello", "who are you"]
    if question.lower() in greetings:
        return ["Hello! I am FinGenie, your AI accountant. I specialize in answering finance-related questions such as tax laws, capital gains, and income finance."]
    
    question_embedding = model.encode(question, convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(np.array([question_embedding]), top_k)

    responses = [chunks[idx] for dist, idx in zip(distances[0], indices[0]) if dist >= similarity_threshold]

    if not responses:
        return ["Please ask something related to finance, tax, or accounting."]

    summarized_responses = [
        summarizer(response, max_length=1000, min_length=30, do_sample=False)[0]['summary_text']
        for response in responses
    ]
    return summarized_responses

# Step 6: Collect and Save Feedback
def save_feedback(question, correct_response, feedback_log="feedback_log.pkl"):
    feedback = {"timestamp": str(datetime.datetime.now()), "question": question, "correct_response": correct_response}
    with open(feedback_log, "ab") as f:
        pickle.dump(feedback, f)

# Step 7: Fine-Tune the Model Using Feedback
def fine_tune_model(feedback_log="feedback_log.pkl", model_name="sentence-transformers/all-mpnet-base-v2"):
    feedback_data = []
    with open(feedback_log, "rb") as f:
        while True:
            try:
                feedback_data.append(pickle.load(f))
            except EOFError:
                break

    examples = [InputExample(texts=[item["question"], item["correct_response"]]) for item in feedback_data]
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)

    model = SentenceTransformer(model_name)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    model.save("fine_tuned_model")
    return model

# Step 8: Validate PDF Paths
def validate_pdf_paths(pdf_paths):
    valid_paths = [path for path in pdf_paths if os.path.exists(path) and path.endswith('.pdf')]
    if not valid_paths:
        raise FileNotFoundError("No valid PDF files found.")
    return valid_paths

# Step 9: Initialize Chatbot Components
pdf_paths = validate_pdf_paths(["data.pdf", "data1.pdf", "data2.pdf", "data3.pdf", "data4.pdf", "data5.pdf", "data6.pdf", "data7.pdf", "data9.pdf", "data10p2.pdf", "data21.pdf", "data12.pdf", "data13.pdf", "data1u6.pdf", "data2u4.pdf"])
raw_text = extract_text_from_pdf(pdf_paths)
chunks = preprocess_text(raw_text)
model_name = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name)

index, chunk_list = load_faiss_index()
if index is None or chunk_list is None:  # Handle missing index file
    print("FAISS index not found. Recreating index from scratch...")
    index, _, chunk_list = create_index(chunks, model_name)
    save_faiss_index(index, chunk_list)


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("question", "")
    if not user_input:
        return jsonify({"error": "No question provided"}), 400
    responses = query_chatbot(user_input, index, model, chunk_list, summarizer)
    return jsonify({"responses": responses, "feedback_prompt": "Was this response correct? (Yes/No)"})

@app.route("/save_chat", methods=["POST"])
def save_chat():
    data = request.json
    chat_name = data.get("chat_name")
    chat = data.get("chat")

    chat_file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_name}.json")

    # If file exists, load and append new messages
    if os.path.exists(chat_file_path):
        with open(chat_file_path, "r") as f:
            existing_chat = json.load(f)
        existing_chat.extend(chat)
        chat = existing_chat  # Update chat history

    with open(chat_file_path, "w") as f:
        json.dump(chat, f, indent=4)

    return jsonify({"message": f"Chat {chat_name} saved successfully!"})

@app.route("/list_chats", methods=["GET"])
def list_chats():
    chats = [f.replace(".txt", "") for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith(".txt")]
    return jsonify({"chats": chats})


@app.route("/load_chat", methods=["POST"])
def load_chat():
    data = request.json
    chat_name = data.get("chat_name")
    chat_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_name}.txt")

    if not os.path.exists(chat_path):
        return jsonify({"error": "Chat not found"}), 404

    with open(chat_path, "r") as file:
        chat = json.load(file)

    return jsonify({"chat": chat})
if __name__ == "__main__":
    app.run(debug=True)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    question = data.get("question", "")
    response = data.get("response", "")
    correct_response = data.get("correct_response", "")
    save_feedback(question, response, correct_response)
    fine_tune_model()
    return jsonify({"message": "Thank you for your feedback! The model is learning from your input."})

if __name__ == "__main__":
    print("FinGenie AI Accountant is now running. Ask me anything about finance!")
    app.run(debug=True)

