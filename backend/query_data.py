import chromadb
import cv2
import os
from google import genai
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from utils import DB_PATH, ImageEmbeddings, DATA_PATH
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def classify_img(client_db, query_img):
    collection = client_db.get_or_create_collection(name="imgs")
    embeddingFunction = ImageEmbeddings()
    
    # Check if the incoming argument is a file string path or an active matrix image
    if isinstance(query_img, str):
        img_matrix = cv2.imread(query_img)
        if img_matrix is None:
            raise FileNotFoundError(f"Could not load local image path target: {query_img}")
    else:
        img_matrix = query_img

    # Send the concrete image matrix to the embedding calculations
    embeddings = embeddingFunction([img_matrix])
    
    results = collection.query(
        query_embeddings=embeddings,
        n_results=1
    )
    return results['ids'][0][0].split('-')[0]


def get_most_similar_chunks(client_db, query_question, img_category):
    collection = client_db.get_collection(name=f"documents_{img_category}")

    results = collection.query(
        query_texts=[query_question],
        n_results=3
    )
    return results['documents'][0], results['metadatas'][0]

def create_response(chunks_text, chunks_metadata, query_question):
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Question: {question}

    """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    context = '\n---\n'.join(chunks_text)
    prompt_string = prompt_template.format(context=context, question=query_question)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    response = client.chat.completions.create(
        model="groq/compound-mini",  # Double-check your model name identifier in Groq console!
        messages=[{"role": "user", "content": prompt_string}]
    )

    return response.choices[0].message.content, chunks_metadata


if __name__ == "__main__":
    # input data: query img, query text
    query_img = 'C:\\Users\\ACER\\Documents\\RAG_APPLICATION\\data\\terrestrial_planets\\earth.png' 
    query_question = 'tell me something about this image?' 

    # create chroma db client
    client_db = chromadb.PersistentClient(path=DB_PATH)

    # classify image
    img_category = classify_img(client_db, query_img)

    # get most similiar chunks
    chunks_text, chunks_metadata = get_most_similar_chunks(client_db, query_question, img_category)

    # create response
    response, sources = create_response(chunks_text, chunks_metadata, query_question)
    print(response)