import os
import numpy as np
import faiss
import pickle
from openai import OpenAI
from dotenv import load_dotenv
from gensim.parsing.preprocessing import remove_stopwords

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

index_path = os.path.join(BASE_DIR, "constitution.index")
chunks_path = os.path.join(BASE_DIR, "chunks.pkl")
metadata_path = os.path.join(BASE_DIR, "metadata.pkl")

index = faiss.read_index(index_path)

with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# def retrieve_chunks(question, k=12, threshold = 0.25):
    # # step 1: embed the question
    # response = client.embeddings.create(
    #         model="text-embedding-3-small",
    #         input=question
    #     )
    # vector = response.data[0].embedding

    # # step 2: convert to numpy array shape (1, 1536)
    # question_array = np.array([vector], dtype=np.float32)

    # # step 2.5: Normalize for cosine similarity
    # faiss.normalize_L2(question_array) 

    # # step 3: search FAISS
    # distances, indices = index.search(question_array, k)

    # # step 4: filter chunks by threshold !
    # results = []
    # for distance, i in zip(distances[0], indices[0]):
    #     if distance > threshold:
    #         results.append({
    #             "chunk": chunks[i],
    #             "metadata": metadata[i],
    #             "score": round(float(distance), 3)
    #         })
            
    
    # return results


def retrieve_chunks(question, k=12, threshold=0.25):
    # clean version
    cleaned = expand_query(question)
    
    # enriched version  
    enriched = enrich_query(cleaned)
    
    # embed both!
    def get_vector(text):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        vector = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        return vector
    
    cleaned_vector = get_vector(cleaned)
    enriched_vector = get_vector(enriched)
    
    # search with both vectors
    d1, i1 = index.search(cleaned_vector, k)
    d2, i2 = index.search(enriched_vector, k)
    
    # combine results, remove duplicates
    seen = set()
    results = []
    
    for dist_arr, idx_arr in [(d1, i1), (d2, i2)]:
        for distance, i in zip(dist_arr[0], idx_arr[0]):
            if i not in seen and distance > threshold:
                seen.add(i)
                results.append({
                    "chunk": chunks[i],
                    "metadata": metadata[i],
                    "score": round(float(distance), 3)
                })
    
    # sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results


def expand_query(question):
    # step 1: lowercase the question
    question = question.lower()

    # step 2: remove filler phrases
    # hint: question.replace("what is", "")
    # remove: "what is", "what are", "how is",
    #         "tell me about", "explain", "define"
    stop_phrases = ["what is", "what are", "how is", "tell me about", "explain", "define"]
    for phrase in stop_phrases:
        question = question.replace(phrase, "")
    # step 3: strip extra spaces
    # hint: question.strip()
    question = question.strip()
    # also remove leading "the", "a", "an"
    clean_question = remove_stopwords(question)

    # step 4: return cleaned question
    return enrich_query(clean_question)

def enrich_query(clean_question):
    # call GPT to generate related legal terms
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a legal search assistant.
                Given a question about Nepal's Constitution,
                generate 5-8 related legal terms or phrases
                that might appear in the Constitution.
                Return ONLY the terms, comma separated.
                No explanation, no sentences."""
            },
            {
                "role": "user",
                "content": f"Question: {clean_question}"
            }
        ],
        max_tokens=100
    )
    related_terms = response.choices[0].message.content

    # combine original + related terms
    enriched = f"{clean_question} {related_terms}"
    return enriched    

def is_followup_question(question):
    """
    Detect if question is a follow-up
    that doesn't need new retrieval
    """
    followup_phrases = [
        "tell me more",
        "more about that",
        "explain more",
        "elaborate",
        "what else",
        "and what about",
        "tell me more about it",
        "can you explain further",
        "go on",
        "continue"
    ]
    question_lower = question.lower()
    return any(phrase in question_lower 
               for phrase in followup_phrases)

def get_original_question(history):
    """
    Walk back through history to find
    the last REAL (non-followup) question
    """
    for msg in reversed(history):
        if msg["role"] == "user":
            if not is_followup_question(msg["content"]):
                return msg["content"]  # found real question!
    return ""  # no real question found

# print(retrieve_chunks())


# if __name__ == "__main__":
#     print (retrieve_chunks("Internation relations?"))

#     question = "What are the fundamental rights of citizens?"
#     results = retrieve_chunks(question)
    
#     print(f"Found {len(results)} relevant chunks\n")
#     for i, result in enumerate(results):
#         print(f"--- Chunk {i+1} ---")
#         print(f"Title: {result['metadata']['title']}")
#         print(f"Score: {result['score']}")
#         print(f"Preview: {result['chunk'][:200]}")
#         print()