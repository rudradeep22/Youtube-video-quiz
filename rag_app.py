import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import subprocess
from groq import Groq
from pytubefix import YouTube
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnablePassthrough
import warnings
from tqdm import tqdm

load_dotenv()
warnings.filterwarnings("ignore")

st.title("YouTube Video Q&A")
VID_LINK = st.text_input("Enter the YouTube link of the video and press Enter:")

# Setting up model
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
parser = StrOutputParser()
template = [
    (
        "system",
        "You are a question answering machine that answers questions based on the following transcription of a video. \n Transcription : {context}"
    ),
    (
        "human",
        "{input}"
    )
]
prompt = ChatPromptTemplate.from_messages(template)

# Function to split audio into chunks
def split_audio(file_path, output_dir, segment_length=60):
    segment_pattern = os.path.join(output_dir, "chunk_%03d.m4a")
    subprocess.run([
        "ffmpeg",
        "-i", file_path,
        "-f", "segment",
        "-segment_time", str(segment_length),
        "-ar", "16000",
        "-ac", "1",
        segment_pattern
    ], check=True)
    return [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.startswith("chunk_")]

# Function to transcribe video
def transcribe_video(link):
    client = Groq()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download audio from YouTube
        youtube = YouTube(link)
        st.write("Downloading audio from YouTube...")
        audio = youtube.streams.get_audio_only()
        file = audio.download(output_path=tmpdir)
        file_path = os.path.join(tmpdir, file)
        
        # Convert audio to 16kHz and mono
        st.write("Converting audio to 16kHz and mono...")
        reduced_file_path = os.path.join(tmpdir, "reduced_audio.m4a")
        subprocess.run([
            "ffmpeg",
            "-i", file_path,
            "-ar", "16000",
            "-ac", "1",
            "-map", "0:a:",
            reduced_file_path
        ], check=True)
        
        # st.write("Splitting audio into chunks...")
        chunks = split_audio(reduced_file_path, tmpdir, segment_length=60)
        
        # Transcribe audio chunks
        st.write("Transcribing audio chunks...")
        transcription_text = ""
        progress_bar = st.progress(0)
        for idx, chunk_path in enumerate(tqdm(chunks, desc="Transcribing", unit="chunk")):
            with open(chunk_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(chunk_path, file.read()),
                    model="distil-whisper-large-v3-en",
                    response_format="json",
                    language="en",
                    temperature=0.0
                )
                transcription_text += transcription.text + "\n"
            progress_bar.progress((idx + 1) / len(chunks))
        
        # Save transcription to file
        with open("transcription.txt", "w") as f:
            f.write(transcription_text)
        
        st.write("Transcription complete.")
        return transcription_text

# Transcription section
if VID_LINK and not os.path.exists("transcription.txt"):
    st.write("Starting transcription process...")
    transcribe_video(VID_LINK)

# Proceed if transcription is available
if os.path.exists("transcription.txt"):
    st.write("Loading transcription...")
    loader = TextLoader("transcription.txt")
    text_documents = loader.load()

    # st.write("Splitting text into chunks for retrieval...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    documents = text_splitter.split_documents(text_documents)

    # st.write("Creating embeddings for document chunks...")
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    hf = HuggingFaceEndpointEmbeddings(
        model=embedding_model,
        task="feature-extraction",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    # st.write("Storing documents into vector store...")
    vectorstore = DocArrayInMemorySearch.from_documents(documents, hf)
    chain = (
        {"context": vectorstore.as_retriever(), "input": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    st.success("Transcription and setup complete! You can now ask questions about the video.")
    st.header("Ask questions about the video!")
    if "query" not in st.session_state:
        st.session_state.query = ""
    query = st.text_input('Enter your question:', key="question_input")

    # Process the query when the user presses Enter
    if query:
        st.write("Fetching answer...")
        with st.spinner("Processing..."):
            result = chain.invoke(query)
        st.write(f"**Answer:** {result}")
        st.session_state.query = ""
