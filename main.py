from fastapi import FastAPI
from langchain import OpenAI
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import YoutubeLoader
import os

os.environ["OPENAI_API_KEY"] = ""

app = FastAPI()

@app.get("/transcribe/{url}")
async def transcribe_video(url: str):
    # Extract the YouTube ID using YoutubeLoader
    loader_youtube = YoutubeLoader.from_youtube_url(url, add_video_info=True, language=["en", "id"])
    docs_youtube = loader_youtube.load()
    youtube_id = docs_youtube[0].metadata['source']

    save_dir = "/data"  # my local to save the temporary audio when download it from youtube
    loader_audio = GenericLoader(
        YoutubeAudioLoader([url], save_dir),
        OpenAIWhisperParser()
    )
    docs_audio = loader_audio.load()

    # Specify the folder path and file name
    folder_path = "/Data"

    # Construct the output filename with the YouTube ID
    file_name = f"{youtube_id}.txt"

    # Combine the folder path and file name
    file_path = os.path.join(folder_path, file_name)

    # Extract page_content from each Document object
    docs_content = [doc.page_content for doc in docs_audio]

    # Convert the list of page_content into a single string
    docs_str = '\n'.join(docs_content)

    with open(file_path, "w") as file:
        file.write(docs_str)

    # Confirmation message
    return {"message": f"Output saved successfully as {file_path}"}
