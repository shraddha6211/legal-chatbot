import faiss
import pickle
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
index = faiss.read_index('constitution.index')

with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

# def test_retrieval(question, k=12):
    # print(f"\nQuestion: {question}")
    # print("=" * 60)
    
    # response = client.embeddings.create(
    #     model='text-embedding-3-small',
    #     input=question
    # )
    # vector = response.data[0].embedding
    # question_array = np.array([vector], dtype=np.float32)
    # faiss.normalize_L2(question_array)
    # distances, indices = index.search(question_array, k)

    # for rank, (score, idx) in enumerate(zip(distances[0], indices[0])):
    #     print(f"Rank {rank+1} | Score: {score:.3f} | Chunk {idx} | {chunks[idx][:80]}")
    
    # # check if chunk 327 appears
    # if 327 in indices[0]:
    #     rank = list(indices[0]).index(327) + 1
    #     print(f"\nChunk 327 found at rank {rank}!")
    # else:
    #     print("\nChunk 327 NOT in top 12!")

# test questions
# test_retrieval("What is the marriage act in Nepal?")
# test_retrieval("What is the right to equality?")
# test_retrieval("Who is the head of state?")

# from retriever import retrieve_chunks

# print("=== Test: Fundamental Rights ===")
# results = retrieve_chunks("What is court marriage?")
# print(f"Chunks found: {len(results)}")
# for r in results:
#     print(f"Score: {r['score']} | Title: {r['metadata']['title']}")

# from retriever import expand_query
# def test_query():
#     questions = [
#         "president election",
#         "fundamental rights",
#         "court marriage nepal"
#     ]

#     for q in questions:
#         expanded = expand_query(q)
#         print(f"Original:  '{q}'")
#         print(f"Enriched:  '{expanded}'")
        
# test_query()
# add this to test.py:
from retriever import retrieve_chunks

questions = [
    "what is right to equality?",
    "how is the president elected?",
    "what are fundamental rights?"
]

for q in questions:
    results = retrieve_chunks(q)
    print(f"Question: '{q}'")
    print(f"Chunks found: {len(results)}")
    for r in results[:3]:
        print(f"  Score: {r['score']} | {r['metadata']['title'][:60]}")
    print()