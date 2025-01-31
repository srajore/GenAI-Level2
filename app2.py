import boto3
import json
import chromadb
import uuid

chroma=chromadb.PersistentClient(path="./chroma_db")

collection = chroma.get_or_create_collection(name="my_collection")


client = boto3.client(service_name='bedrock-runtime')

# Prepare the request body with proper JSON structure
request_body = {
    "inputText": "Describe the purpose of 'hello world' program in one line"
}

response = client.invoke_model(
    modelId='amazon.titan-embed-text-v2:0',
    body=json.dumps(request_body),  # Convert dict to JSON string
    contentType='application/json',
    accept='application/json'
)


# parse the response 

embeddings_data = json.loads(response['body'].read().decode('utf-8'))

embeddings = embeddings_data['embedding']


unique_id = str(uuid.uuid4())


collection.add(
    documents=[request_body['inputText']],
    embeddings=embeddings,
    ids=[unique_id],
    metadatas=[{"source": "user_input"}]
)


print("Embeddings added to ChromaDB successfully!")

# Query to the collection

query_result =collection.query(
    query_embeddings=[embeddings],
    n_results=1,
    include=["documents", "embeddings", "distances"]
)


# Convert query results to JSON-serializable format
# serializable_result = {
#     "documents": query_result["documents"],
#     "embeddings": [emb.tolist() for emb in query_result["embeddings"]],
#     "distances": query_result["distances"]
# }

# print(json.dumps(serializable_result, indent=2))



print("Query results:")


print(query_result)
