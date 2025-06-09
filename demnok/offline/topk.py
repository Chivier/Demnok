import faiss
import json
import asyncio
import aiohttp
import numpy as np
from typing import List, Dict, Any, Optional

async def get_embedding(session: aiohttp.ClientSession, text: str, base_url: str) -> List[float]:
    """Get embedding for a single text using async HTTP request."""
    url = f"{base_url}/embeddings"
    payload = {
        "model": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "input": text,
    }
    
    async with session.post(url, json=payload) as response:
        result = await response.json()
        return result["data"][0]["embedding"]

async def process_query(session: aiohttp.ClientSession, index: int, qid: str, ans: List[str], 
                       question_with_title: str, indices: faiss.Index, base_url: str) -> tuple:
    """Process a single query: get embedding and search index. Returns (index, result)."""
    try:
        embedding = await get_embedding(session, question_with_title, base_url)
        
        # Convert to numpy array and reshape for faiss
        embedding_array = np.array([embedding], dtype=np.float32)
        
        # Search in faiss index
        D, I = indices.search(embedding_array, 20)
        
        result = {
            "qid": qid,
            "text": question_with_title,
            "answer": ans,
            "top_k_doc_id": [int(i) for i in I[0]]
        }
        
        return (index, result)
    except Exception as e:
        print(f"Error processing query {qid} at index {index}: {e}")
        return (index, None)

async def process_queries_concurrent(queries_data: List[tuple], indices: faiss.Index, 
                                   base_url: str, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """Process all queries concurrently while preserving order."""
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(index_and_query_data):
            async with semaphore:
                index, (qid, ans, question_with_title) = index_and_query_data
                return await process_query(session, index, qid, ans, question_with_title, indices, base_url)
        
        # Create tasks for all queries with their original indices
        indexed_queries = [(i, query_data) for i, query_data in enumerate(queries_data)]
        tasks = [process_with_semaphore(indexed_query) for indexed_query in indexed_queries]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Create a list to store results in order
        ordered_results = [None] * len(queries_data)
        
        # Place results in their original positions
        for result in results:
            if not isinstance(result, Exception) and result is not None:
                index, data = result
                if data is not None:
                    ordered_results[index] = data
        
        # Filter out None results while maintaining order
        return [r for r in ordered_results if r is not None]

async def main():
    query_path = "datasets/nrtvQA_new/nrtvQA_new_queries.jsonl"
    index_path = "datasets/nrtvQA_new/nrtvQA_new_index.faiss"
    port = 30000
    base_url = f"http://localhost:{port}/v1"
    
    # Load faiss index
    indices = faiss.read_index(index_path)
    
    # Load and prepare data
    with open(query_path, "r") as f:
        data = [json.loads(line) for line in f]
        questions = [item["question"] for item in data]
        answers = [item["answers"] for item in data]
        titles = [item["title"] for item in data]
        qids = [item["qid"] for item in data]
    
    questions_with_titles = [f"{question} in '{title}'" for title, question in zip(titles, questions)]
    
    # Prepare data for concurrent processing
    queries_data = list(zip(qids, answers, questions_with_titles))
    
    print(f"Processing {len(queries_data)} queries...")
    
    # Process queries concurrently while preserving order
    new_queries = await process_queries_concurrent(queries_data, indices, base_url, max_concurrent=10)
    
    print(f"Successfully processed {len(new_queries)} queries")
    
    # Save results (now in original order)
    with open("datasets/nrtvQA_new/nrtvQA_new_queries_top20.jsonl", "w") as f:
        for item in new_queries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    asyncio.run(main())

