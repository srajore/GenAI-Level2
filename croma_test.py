import chromadb

try:
    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(name="my_collection")
 
    collection.upsert(
        documents=[
            "This is a document about pineapple",
            "This is a document about oranges"
        ],
        ids=["id1", "id2"]
    )
  
    results = collection.query(
        query_texts=["This is a query document about florida"],
        n_results=2
    )
    
    print("Query results:", results)
    print("Number of results:", len(results['documents'][0]) if results else 0)

except Exception as e:
    print(f"An error occurred: {str(e)}")
