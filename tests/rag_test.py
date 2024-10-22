from demnok.agents import HFChatAgent, HFInstructEmbeddingAgent
from demnok.engines import QdrantRAGEngine
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff
import torch
from demnok.utils import SimpleTextChunker

data = "./data/cs_extractions/serverlessllm/serverlessllm.md"
chunker = SimpleTextChunker()
text = chunker.file_read(data)
chunks = chunker.langchain_chunk(text)

embedding_agent = HFInstructEmbeddingAgent(
    model_name="Alibaba-NLP/gte-large-en-v1.5",
    torch_dtype=torch.float32,
    chunks=chunks
    )

chat_agent = HFChatAgent(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16
    )

client = QdrantClient(path="/home/ubuntu/qdrant-store")

collection_name = "serverlessllm"
embedding_store_name = "embedding_lst_serverlessllm.pkl"

if not client.collection_exists(collection_name):
    client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.EUCLID),
    hnsw_config=HnswConfigDiff(
        m=16, 
        ef_construct=100,
        full_scan_threshold=10000
        )
    )

embedding_lst = embedding_agent.get_corpus_embeddings(embedding_store_name, max_length=8192)

client.upsert(
    collection_name=collection_name,
    points=[
        PointStruct(
            id=i,
            vector=embedding
        )
        for i, embedding in enumerate(embedding_lst)
    ]
)

engine = QdrantRAGEngine(embedding_agent, client, chat_agent, collection_name=collection_name)

answer = engine.rag_cot("What is live migration in serverlessllm?", 100)

print(answer[0])

# chain.extend(answer)

# print("\n\n".join(chain))

# questions = [
#     '''
# Address the following question in a step-by-step manner: What is the capital of France?

# IMPORTANT: 
# - Structure each thought in paragraphs, ensuring each is clearly separated by ##. 
# - Refrain from using digits or numbered lists. Limit the response to a maximum of 4 paragraphs.
#     '''
# ]

# ans = agent.chat(questions)

# print(ans[0])