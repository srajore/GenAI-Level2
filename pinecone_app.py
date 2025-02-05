import boto3
from pinecone import Pinecone,ServerlessSpec

from dotenv import load_dotenv # Import the load_dotenv function from the dotenv module
import os
import time
import json

# Load the environment variables from the .env file
load_dotenv()

# Initialize the boto3 client
bedrock_client = boto3.client('bedrock-runtime')

# Initialize the Pinecone client
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),  # Get the Pinecone API key from the environment variables
)

# Name of the Pinecone index
index_name = "test-index"


try:
    # Check if the index already exists in Pinecone
    if index_name not in pc.list_indexes().names():
        # If the index doesn't exist, create it
        pc.create_index(
            name=index_name,
            dimension=1536,  # Dimension of the embeddings
            metric="cosine", # Distance metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)  # Wait for the index to be ready

    index=pc.Index(index_name)

    # Function to generate embeddings using the Bedrock API Titan Embed Text model
    def get_embeddings(text):
        # Call Bedrock API to get embeddings
        response = bedrock_client.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            body=json.dumps({'inputText': text}),
            contentType='application/json',
            accept='application/json'
        )
        # Extract and return the embedding vector
        return json.loads(response['body'].read())['embedding']
    
    # Sample text for which we want to generate embeddings
    texts=[
        "Hello wrold program displays Hello world message",
        "Python is a programming language",
        "Machine learning is facinating"
    ]

    # store each text and its embedding in the Pinecone
    for i , text in enumerate(texts):
        # Generate embeddings for the text
        embeddings=get_embeddings(text)

        # Store the text and its embedding in Pinecone
        index.upsert(vectors=[{
            'id': str(i),  # Use the index as the ID
            'values': embeddings, # Embedding vector
            'metadata': {'text': text} #original text
        }])

        print(f"Stored text {i+1}/{len(texts)} in Pinecone.")


    # Query the index with a sample text
    query_text = "Machine learniing"

    query_embeddings = get_embeddings(query_text)

    search_results = index.query(
        vector=query_embeddings,  # Query vector
        top_k=1,  # Return top 1 most similar results
        include_metadata=True # Include metadata in the results
    )

    #Display the search results
    print("\n Search Results:")

    for match in search_results['matches']:
        print(f"Text: {match['metadata']['text']}")  # Original text
        print(f"Score: {match['score']}\n")  # Similarity score(HIGHER IS THE BETTER)
        print(" -----    ")


except Exception as e:
    print(f"An error occurred: {e}")


