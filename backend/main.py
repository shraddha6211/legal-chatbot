from fastapi import FastAPI
from pydantic import BaseModel
from retriever import retrieve_chunks, is_followup_question, get_original_question
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

# define request and response shapes
class ChatRequest(BaseModel):
    question: str
    history: list = []

class ChatResponse(BaseModel):
    answer: str
    sources: list

# @app.post("/chat", response_model=ChatResponse)
# def chat(request: ChatRequest):
#     rel_chunks = []
#     # step 1: retrieve relevant chunks
#     # hint: use retrieve_chunks(request.question)
#     chunks = retrieve_chunks(request.question)  # question extracts just the text string

#     # step 2: combine chunks into one context string
#     # hint: "\n\n".join(chunks) joins list into one string
#     context = "\n\n".join(chunks)

#     # step 3: build the prompt
#     # hint: use the prompt template above
#     # put context and request.question inside it
#     prompt = f"""
# You are a legal assistant expert on the Constitution of Nepal.
# Answer the question using ONLY the context provided below.
# Combine information from ALL context sections to form 
# a complete answer.
# If the answer is not in the context, say exactly what 
# you CAN find and suggest the user search for specific 
# article numbers.
# Always cite article numbers when mentioned in context.

# Context:
# {context}

# Question: {request.question}

# Answer in a clear, structured way.
# """

#     # step 4: call GPT
#     # hint: client.chat.completions.create(...)
#     response = client.chat.completions.create(
#         model="gpt-4.1-nano",
#         messages=[
#             {"role": "system", "content": "You are a legal assistant"},
#             {"role": "user", "content": prompt}
#         ]
#     )

#     # step 5: extract answer from response
#     # hint: response.choices[0].message.content
#     answer = response.choices[0].message.content

#     # step 6: return ChatResponse
#     # hint: return ChatResponse(answer=..., sources=...)
#     return ChatResponse(answer=answer, sources=chunks)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    print("=== CHAT CALLED ===")
    print(f"Question: {request.question}")
    print(f"History length: {len(request.history)}")
    
    try:
        # results = retrieve_chunks(request.question)
        # print(f"Chunks retrieved: {len(results)}")
        
        # if not results:
        #     print("WARNING: No chunks returned!")
        #     return ChatResponse(
        #         answer="No relevant information found in the Constitution.",
        #         sources=[]
        #     )
        # check if follow-up question
        if is_followup_question(request.question) and request.history:
            print("Follow-up detected! Reusing previous context...")
            original = get_original_question(request.history)
            print(f"Original question: {original}")
            results = retrieve_chunks(original) if original else []
        else:
            results = retrieve_chunks(request.question)

        print(f"Chunks retrieved: {len(results)}")
            
        if results:
            print("First chunk preview:")
            print(results[0]['chunk'][:200])

        if not results:
            return ChatResponse(
                answer="I cannot find this in the Constitution of Nepal. Please consult relevant Nepal legislation or a legal professional.",
                sources=[]
            )

        context = "\n\n".join([r["chunk"] for r in results])
        print(f"Context length: {len(context)}")

        sources = [
            f"{r.get('metadata', {}).get('title', 'Unknown')} (score: {r.get('score', 0)})"
            for r in results
        ]

#             # get previous question from history
#             prev_question = ""
#             for msg in reversed(request.history):
#                 if msg["role"] == "user":
#                     prev_question = msg["content"]
#                     break
            
#             print(f"Previous question: {prev_question}")
            
#             # search using previous question!
#             results = retrieve_chunks(prev_question)
#         else:
#             # normal search with current question

#             results = retrieve_chunks(request.question)
#         print(f"Chunks retrieved: {len(results)}")

#         # add this:
#         if results:
#             print("First chunk preview:")
#             print(results[0]['chunk'][:200])

#         if not results:
#             return ChatResponse(
#                 answer="I cannot find this in the Constitution of Nepal. Please consult relevant Nepal legislation or a legal professional.",
#                 sources=[]
#             )
            
#         context = "\n\n".join([r["chunk"] for r in results])
#         print(f"Context length: {len(context)}")
        
#         # build sources for display
#         sources = [
#             f"{r.get('metadata', {}).get('title', 'Unknown')} (score: {r.get('score', 0)})"
#             for r in results
# ]

        prompt = f"""
You are a legal assistant for the Constitution of Nepal.
Answer using ONLY the context provided below.

RULES:
1. Context contains relevant articles → summarize and cite them
2. Answer concisely (3-5 sentences) unless specifically asked to be detailed
3. Always mention article numbers found in context
4. Only say "not found" if context is completely empty
   or truly irrelevant to the question
5. NEVER use knowledge outside the context
6. If context is related but not exact → still answer from it!
7. Use conversation history to understand follow-up questions

Context:
{context}

Question: {request.question}

""" 
         # build messages with history!
        messages = [
            {"role": "system", "content": "You are a legal assistant for the Constitution of Nepal. Always answer from provided context. Use conversation history to understand follow-up questions."}
        ]
        
        # add last 10 messages from history
        for msg in request.history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # add current question with context
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        print(f"Total messages sent to GPT: {len(messages)}")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        answer = response.choices[0].message.content
        print(f"Answer: {answer[:100]}")
        
        return ChatResponse(answer=answer, sources=sources)
        
#         print("Calling GPT...")
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a legal assistant for the Constitution of Nepal. Always answer from the provided context. Never say not found if context contains related information."},
#                 {"role": "user", "content": prompt}
# ]
#         )
        
#         answer = response.choices[0].message.content
#         print(f"Answer received: {answer[:100]}")
        
#         return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        
        print(f"ERROR: {str(e)}")
        return ChatResponse(
            answer=f"Error: {str(e)}",
            sources=[]
        )

        # print(f"ERROR: {str(e)}")
        # return ChatResponse(
        #     answer=f"Error: {str(e)}",
        #     sources=[]
        # )