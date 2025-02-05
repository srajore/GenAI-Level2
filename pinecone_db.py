
from dotenv import load_dotenv
import os
from pinecone import Pinecone

import os

load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("genai")


print(index.describe_index_stats())

