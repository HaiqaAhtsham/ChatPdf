import streamlit as st
from deepgram import Deepgram
import asyncio
import fitz
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import os
import torch
torch.classes.__path__ = []
from transformers import GPT2Tokenizer

# Replace with your Deepgram API key
DEEPGRAM_API_KEY = '10f461e787d5fb636a63338974b362eb8331b742'
dg_client = Deepgram(DEEPGRAM_API_KEY)
# Set your Groq API key
os.environ["GROQ_API_KEY"] = "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"

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Initialize the ChatGroq model
chat_model = ChatGroq(model="llama-3.3-70b-versatile") 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to transcribe audio using Deepgram
async def transcribe_audio(file):
    source = {'buffer': file, 'mimetype': 'audio/wav'}
    response = await dg_client.transcription.prerecorded(source, {'punctuate': True})
    transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
    return transcript

def extract_pdf_text(pdf_file):
    pdf_reader = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ''
    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        text += page.get_text()
    return text
def split_text_into_chunks(text, max_tokens=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence))
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    return chunks

def generate_answer(pdf_chunks, query):
    answers = []
    for chunk in pdf_chunks:
        prompt = f"Document:\n{chunk}\n\nQuestion:\n{query}\n\nAnswer:"
        response = chat_model.invoke(prompt)
        answers.append(response)
    return ' '.join(answers)

# Streamlit app
st.title("Chat with Your Doc")

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
if pdf_file:
    pdf_text = extract_pdf_text(pdf_file)
    st.success("PDF uploaded and processed successfully.")

    # Record audio question
    audio_data = st.audio_input("Record your question")
    if audio_data:
        st.audio(audio_data, format='audio/wav')
        if st.button("Transcribe and Get Answer"):
            with st.spinner("Transcribing audio..."):
                transcript = asyncio.run(transcribe_audio(audio_data))
                st.write(f"**Transcribed Question:** {transcript}")

                # Generate answer using ChatGroq
                with st.spinner("Generating answer..."):
                    answer = generate_answer(pdf_text, transcript)
                    st.write(f"**Answer:** {answer}")