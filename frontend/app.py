import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Nepal Constitution Chatbot",
    page_icon = "🏛️"
)

# title
st.title("🏛️ Nepal Constitution Legal Chatbot")
st.caption("Ask anything about the Constitution of Nepal")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# chat input at bottom
question = st.chat_input("Ask a legal question...")

if question:
    # step 1: display user message
    # hint: st.chat_message("user") + st.write(question)
    # hint: append to st.session_state.messages
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # step 2: send to backend
    # hint: requests.post("http://127.0.0.1:8000/chat", json=...)
    try:
        with st.spinner("Searching the Constitution..."):
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json = {
                        "question": question,
                        "history": st.session_state.messages[:-1]
                        # [:-1] excludes current question
                        # already added above!
                    }
            )
            data = response.json()

        # step 3: display bot answer
        # hint: st.chat_message("assistant") + st.write(answer)
        with st.chat_message("assistant"):
            st.write(data["answer"])

        # step 4: display sources
        # hint: st.expander("📚 Sources") to make collapsible section
        with st.expander("📚 Sources"):
            if data["sources"]:
                for i, source in enumerate(data["sources"]):
                    st.write(f"**{i+1}.** {source}")
            else:
                st.info("No specific articles found.")
        # step 5: append bot response to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["answer"]
        })
    
    except Exception as e:
        st.error(f"⚠️ Could not connect to backend: {str(e)}")
        st.info("Make sure FastAPI server is running!")