import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from retriever import retrieve_chunks, is_followup_question, get_original_question
from openai import OpenAI
from dotenv import load_dotenv


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

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_constitution",
            "description": "Search the Constitution of Nepal for relevant articles and sections. Use this when user asks about constitutional rights, duties, government structure, elections, or any legal topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant constitutional articles"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search conversation history for relevant past questions and answers. Use when user refers to previous conversation, asks follow-up questions, says 'tell me more', 'what did I ask', 'earlier', 'before', 'that', 'it'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in conversation history"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_article",
            "description": "Fetch a specific article from the Constitution by article number. Use when user mentions a specific article number like 'Article 18' or 'Article 42'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "article_number": {
                        "type": "integer",
                        "description": "The article number to fetch from Constitution"
                    }
                },
                "required": ["article_number"]
            }
        }
    }
]

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    print("=== CHAT CALLED ===")
    print(f"Question: {request.question}")
    print(f"History length: {len(request.history)}")

    # history lives here in father's scope!
    history = request.history
    sources = []

    # ── INNER TOOL FUNCTIONS (sons!) ──────────────────

    def search_constitution(query: str):
        print(f"Tool called: search_constitution('{query})")
        results = retrieve_chunks(query)
        if not results:
            return "No relevant articles found in the constitution."
        
        # save sources for display
        for r in results:
            title = r.get('metadata', {}).get('title', 'Unknown')
            score = r.get('score', 0)
            source = f"{title} (score: {score})"
            if source not in sources:
                sources.append(source)

        # return chunks as text
        context = "\n\n".join([r["chunk"] for r in results])
        print(f"Constitution search returned {len(results)} chunks")
        return context
    
    def search_memory(query: str):
        print(f"Tool called: search_memory('{query}')")

        if not history:
            return "No conversation history available."
        
        # search history for relevant messages
        relevant = []
        query_lower = query.lower()
        
        for msg in history:
            if any(word in msg["content"].lower()
                   for word in query_lower.split()):
                relevant.append(
                    f"{msg['role'].upper()}: {msg['content']}"
                )
        
        if not relevant:
            # return last 4 messages if no match
            recent = history[-4:]
            relevant = [
                f"{m['role'].upper()}: {m['content']}"
                for m in recent
            ]

        result = "\n".join(relevant)
        print(f"Memory search returned {len(relevant)} messages")
        return result

    def get_article(article_number: int):
        print(f"Tool Called: get_article({article_number})")

        # search for specific article number
        results = retrieve_chunks(f"Article {article_number}")

        for r in results:
            chunk = r["chunk"]
            # check if this chunk contains our article
            if str(article_number) in chunk[:50]:
                title = r.get('metadata', {}).get('title', 'Unknown')
                sources.append(f"{title} (direct lookup)")
                print(f"Article {article_number} found!")
                return chunk
            
        return f"Article {article_number} not found in Constitution."
    
    # ── TOOL DISPATCHER ───────────────────────────────

    def run_tool(tool_name: str, tool_args: dict):
        """
        Runs the correct tool based on GPT's choice
        """
        if tool_name == "search_constitution":
            return search_constitution(tool_args["query"])
        elif tool_name == "search_memory":
            return search_memory(tool_args["query"])
        elif tool_name == "get_article":
            return get_article(tool_args["article_number"])
        else:
            return f"Unknown tool: {tool_name}"
        
    # ── MAIN CHAT LOGIC ───────────────────────────────
    try:
        # build initial messages
        messages = [
            {
                "role": "system",
                "content": """You are a legal assistant for the Constitution of Nepal.
You have access to three tools:
1. search_constitution: search for legal articles
2. search_memory: search conversation history  
3. get_article: fetch specific article by number

RULES:
- Always use tools to find information
- Never answer from your own knowledge
- Cite article numbers in your answer
- Be concise (3-5 sentences) unless asked for detail
- If tools return no results, say "not found in Constitution"
"""
            }
        ]

        # add conversation history
        for msg in history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # add current question
        messages.append({
            "role": "user",
            "content": request.question
        })

        print("Sending to GPT with tools...")

        # ── TOOL CALLING LOOP ─────────────────────────
        # GPT might call multiple tools!
        # We loop until GPT stops calling tools

        while True:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto"    # GPT decides!
            )

            message = response.choices[0].message
            print(f"GPT response type: {response.choices[0].finish_reason}")

            # add GPT response to messages
            messages.append(message)

            # check if GPT wants to call tools
            if response.choices[0].finish_reason == "tool_calls":
                # GPT wants to call one or more tools!
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"GPT calling: {tool_name}({tool_args})")

                    # run the tool
                    tool_result = run_tool(tool_name, tool_args)

                    # send result back to GPT
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # loop again → GPT reads results
                # might call more tools or give final answer!

            else:
                # GPT is done calling tools!
                # This is the final answer
                answer = message.content
                print(f"Final answer: {answer[:100]}")
                break

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return ChatResponse(
            answer=f"Error: {str(e)}",
            sources=[]
        )