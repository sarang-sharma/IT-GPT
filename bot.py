import openai
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models

st.title("IT-GPT")

openai.api_key = st.secrets["OPENAI_API_KEY"]

OPENAI_EMBEDDINGS_ENGINE = "text-embedding-ada-002"


qdrant_client = QdrantClient(
    "https://96a4ea49-33c9-4e25-93c6-98048c7725ee.europe-west3-0.gcp.cloud.qdrant.io", 
    prefer_grpc=True,
    api_key = st.secrets["QDRANT_KEY"],
)


def get_embedding(text: str, model: str = OPENAI_EMBEDDINGS_ENGINE):
    text = text.replace("\n", " ")
    result = openai.Embedding.create(
        engine=model,
        input=text[:28000]
    )
    return result["data"][0]["embedding"]

def total_characters(messages):
    return sum(len(message["content"]) for message in messages)

def add_message(messages, message):
    messages.append(message)
    while total_characters(messages) > 50000:
        messages.pop(0)
    return messages


def retrieve_context_from_qdrant(query_embedding, limit: int = 20):
    search_response = qdrant_client.search(
        collection_name="LLA",
        search_params=models.SearchParams(
            hnsw_ef=256,    
            exact=False
        ),
        query_vector=query_embedding,
        limit=limit,
        with_vectors = False,
        )
    
    scored_context = []
    for result in search_response:
        try:
            if result.score > 0.745:
                scored_context.append((result.score, result.payload.get('text'), result.payload.get('position')))
        except KeyError:
            print("Error: Expected key structure not found in point.")

    # First, sort by distance for relevance
    scored_context.sort(key=lambda x: x[0])

    # Then, sort by position to ensure adjacent sections are next to each other
    scored_context = sorted(scored_context, key=lambda x: x[2])

    # Extract the context sections from the sorted list
    context_sections = [text for _, text, _ in scored_context]

    return context_sections

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-16k"

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "last_context" not in st.session_state:
    st.session_state.last_context = []

# Display previous chat messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
with st.chat_message("assistant"):
    st.markdown("Hi, there! I can help you with any query related to Income Tax.")        

if prompt := st.chat_input("Type your query here..."):

    add_message(st.session_state.messages, {"role": "system", "content": "You are a helpful assistant specializing in simplifying and explaining the intricacies of the Indian income tax act and labour laws of India for someone unfamiliar with these topics. Your responses should be for helping a common man"})

    # Save the user's message to the session state
    add_message(st.session_state.messages, {"role": "user", "content": prompt})

    # Display the latest user input message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    query_embedding = get_embedding(prompt)
    
    context_sections = retrieve_context_from_qdrant(query_embedding)

    # Check if context_sections is empty, and if so, use the last context
    if not context_sections and st.session_state.last_context:
        context_sections = st.session_state.last_context
    else:
        # Save the retrieved context for potential future use
        st.session_state.last_context = context_sections

    # Combine context sections into one system message to prevent exceeding message limit
    combined_context = " ".join(context_sections)
    system_message = {
        "role": "system",
        "content": f"Answer the user's query from the provided context. Give the relevant sections, articles as the sources and links wherever possible. Give examples to supplement your response. Strictly answer only from the context provided.\n\nContext:\n {combined_context[:40000]}."
    }
    
    # This is our working copy of messages for the current completion
    current_messages = st.session_state.messages.copy()
    current_messages.append(system_message)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        print("\n\n\n------------------------\n\n\n", current_messages)
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=current_messages,
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        # print("\n\n\n------------------------\n\n\n", full_response)

    # Add only the assistant's response to the session state
    add_message(st.session_state.messages, {"role": "assistant", "content": full_response})
