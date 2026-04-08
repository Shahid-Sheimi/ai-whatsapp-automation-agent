import uuid
import os

import qdrant_client
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client_connection = QdrantClient(url="http://localhost:6333")

try:
    client_connection.delete_collection(collection_name="lost_package_policy")
except:
    pass
try:
    client_connection.delete_collection(collection_name="shipping_information")
except:
    pass

qdrant_vector_config = qdrant_client.http.models.VectorParams(
    size=1536,
    distance=qdrant_client.http.models.Distance.COSINE,
)


def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


client_connection.create_collection(
    collection_name="lost_package_policy", vectors_config=qdrant_vector_config
)

with open("data/lost_package_policy.md", "r", encoding="utf-8") as f:
    content = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
    length_function=len,
)

chunks = text_splitter.split_text(content)

points = []
for chunk in chunks:
    vector = get_embedding(chunk)
    doc_id = str(uuid.uuid4())
    points.append(PointStruct(id=doc_id, vector=vector, payload={"text": chunk}))

client_connection.upsert(collection_name="lost_package_policy", points=points)

client_connection.create_collection(
    collection_name="shipping_information", vectors_config=qdrant_vector_config
)

with open("data/shipping_information.md", "r", encoding="utf-8") as f:
    content = f.read()

chunks = text_splitter.split_text(content)

points = []
for chunk in chunks:
    vector = get_embedding(chunk)
    doc_id = str(uuid.uuid4())
    points.append(PointStruct(id=doc_id, vector=vector, payload={"text": chunk}))

client_connection.upsert(collection_name="shipping_information", points=points)

print("Vector database created successfully!")
print("- lost_package_policy collection created")
print("- shipping_information collection created")
