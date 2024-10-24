import os
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
from tqdm import tqdm
import warnings

load_dotenv()
warnings.filterwarnings("ignore")

print("If don't want to transcribe a new video, leave it blank.")
VID_LINK = input('Enter the youtube link of the video: ')

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

# Transcribing video
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

print("Starting transcription process...")

if not os.path.exists("transcription.txt"):
    youtube = YouTube(VID_LINK)
    print("Downloading audio from YouTube...")
    audio = youtube.streams.get_audio_only()

    client = Groq()
    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        file_path = os.path.join(tmpdir, file)

        print("Converting audio to 16kHz and mono...")
        reduced_file_path = os.path.join(tmpdir, "reduced_audio.m4a")
        subprocess.run([
            "ffmpeg",
            "-i", file_path,
            "-ar", "16000",
            "-ac", "1",
            "-map", "0:a:",
            reduced_file_path
        ], check=True)

        print("Splitting audio into chunks...")
        chunks = split_audio(reduced_file_path, tmpdir, segment_length=60)
        transcription_text = ""

        print("Transcribing audio chunks...")
        for chunk_path in tqdm(chunks, desc="Transcribing", unit="chunk"):
            with open(chunk_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(chunk_path, file.read()),
                    model="distil-whisper-large-v3-en",
                    response_format="json",
                    language="en",
                    temperature=0.0
                )
                transcription_text += transcription.text + "\n"

        print("Saving transcription to file...")
        with open("transcription.txt", "w") as f:
            f.write(transcription_text)

print("Loading transcription...")
loader = TextLoader("transcription.txt")
text_documents = loader.load()

print("Splitting text into chunks for retrieval...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)

print("Creating embeddings for document chunks...")
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
hf = HuggingFaceEndpointEmbeddings(
    model=embedding_model,
    task="feature-extraction",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

print("Storing documents into vector store...")
vectorstore = DocArrayInMemorySearch.from_documents(documents, hf)

print("Setting up the retrieval chain...")
chain = (
    {"context": vectorstore.as_retriever(), "input": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

print("Ready for questions. Leave question blank to quit")
while True:
    query = input('Enter your question: ')
    if query == "":
        break
    print("Fetching answer...")
    result = chain.invoke(query)
    print(f"Answer : {result}")