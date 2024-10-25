
# YouTube Video Question Answering

This project allows you to perform question answering on YouTube videos by transcribing the video content and using a language model to answer questions based on the transcription.

## Features

- Download audio from YouTube videos.
- Transcribe audio to text.
- Split transcriptions into manageable chunks.
- Create embeddings for document chunks.
- Store documents in a vector store for efficient retrieval.
- Answer questions based on the transcribed content.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment:**

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the root directory of the project and add the following environment variables:

    ```env
    GROQ_API_KEY=<your-groq-api-key>
    HUGGINGFACEHUB_API_TOKEN=<your-huggingfacehub-api-token>
    ```

## Why Transcribe Audio This Way?

The transcription process involves converting audio to 16kHz mono and splitting it into chunks before transcribing. This approach is taken for several reasons:

* **Consistency and Quality:** Converting audio to a standard format (16kHz mono) ensures consistent quality and compatibility with the transcription API.
* **Manageable Chunks:** Splitting the audio into smaller chunks makes the transcription process more manageable and reduces the likelihood of errors or timeouts.

## Usage

1. **Run the Streamlit app:**

    ```sh
    streamlit run rag_app.py
    ```

2. **Provide the YouTube video link:**

    When prompted, enter the YouTube link of the video you want to transcribe and analyze. 

3. **Ask questions:**

    After the transcription process is complete, you can start asking questions based on the transcribed content. Leave the question blank to quit the program.

## How It Works

1. **Download Audio:**
    - The script downloads the audio from the provided YouTube link using `pytubefix`.

2. **Transcribe Audio:**
    - The audio is converted to 16kHz mono and split into chunks.
    - Each chunk is transcribed using the Groq API.

3. **Process Transcription:**
    - The transcription is saved to a file and loaded for further processing.
    - The text is split into chunks for efficient retrieval.

4. **Create Embeddings:**
    - Embeddings for the document chunks are created using a HuggingFace model.

5. **Store Documents:**
    - The documents are stored in a vector store for efficient retrieval.

6. **Answer Questions:**
    - A retrieval chain is set up to answer questions based on the transcribed content using a language model.

## Dependencies

- `langchain`
- `langchain[docarray]`
- `langchain_groq`
- `langchain_community`
- `langchain_huggingface`
- `docarray`
- `pydantic==1.10.8`
- `pytubefix`
- `python-dotenv`
- `tiktoken`
- `ruff`
- `pypdf`
- `groq`
- `streamlit`
