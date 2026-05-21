import os
import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from utils import ImageEmbeddings, DATA_PATH, DB_PATH
import dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

dotenv.load_dotenv()

# new chromaDB client
client_db = chromadb.PersistentClient(path=DB_PATH)
# create images collection
img_collection = client_db.create_collection(
    name="imgs",
    embedding_function=ImageEmbeddings(),
    data_loader=ImageLoader()
)

for dir_ in os.listdir(DATA_PATH):
    dir_path = os.path.join(DATA_PATH, dir_)
    # Add images to the collection
    img_collection.add(
        ids=[f"{dir_}-{img_path}" for img_path in os.listdir(dir_path) if img_path.endswith('.png')], 
        uris=[os.path.join(dir_path, img_path) for img_path in os.listdir(dir_path) if img_path.endswith('.png')],
    )
    
    # collection = client_db.create_collection(
    #     name=f"documents_{dir_}",
    #     embedding_function=OpenAIEmbeddingFunction(
    #         api_key=os.getenv("OPENAI_API_KEY"),
    #         model_name="text-embedding-3-small"
    #     )
    # )
    
    embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

   # Pass this embedding_function when you get or create your collection:
    collection = client_db.create_collection(
        name=f"documents_{dir_}", 
        embedding_function=embedding_function
    )

    # Load Documents
    loader = DirectoryLoader(dir_path, glob="*.txt")
    documents = loader.load()
    
    # converting documents to chunks
    
    text_splitter = RecursiveCharacterTextSplitter(
	        chunk_size=300,
	        chunk_overlap=100,
	        length_function=len,
	        add_start_index=True,
	    )
    
    chunks = text_splitter.split_documents(documents)
    
    #  Add chunks to documents_collection
    
    collection.add(
		ids=[str(j) for j in range(len(chunks))],
		documents=[chunks[j].page_content for j in range(len(chunks))],
		metadatas=[chunks[j].metadata for j in range(len(chunks))],
	)