import os
import streamlit as st
from src.processor import extract_video_id, get_transcript
from src.llm_engine import LearningBot
from src.vector_store import index_single_video
from src.exporter import generate_pdf, export_output
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the bot once
bot = LearningBot(api_key=api_key)

# Initializing session state variable
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = None

if "last_video_id" not in st.session_state:
    st.session_state.last_video_id = None

# load vector database
@st.cache_resource
def load_vector_db(api_key):
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=OpenAIEmbeddings(api_key=api_key)
    )

# We need to make sure we can modify this vector_db instance on the fly
if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_vector_db(api_key=api_key)

# --------------------- SIDEBAR NAVIGATION ----------------#
st.sidebar.title("Naviation")
page = st.sidebar.radio(
    "Go to",
    ["YouTube Learning", "RAG Chatbot"]
)

# ----- Page 1: YouTube Learning -------------#
if page == "YouTube Learning":
    
    st.title("AI Study Buddy")
    
    video_url = st.text_input(
        "Paste YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    if video_url:
        video_id = extract_video_id(video_url)

        if video_id:
            st.success(f"Found Video ID: {video_id}")

            if st.button("Generate Learning Materials"):
                with st.spinner("Reading transcript..."):
                    transcript = get_transcript(video_id)
                    
                if "Error" in transcript:
                    st.error(transcript)
                else:
                    # Set active video session ID
                    st.session_state.active_video_id = video_id

                    # Save transcript
                    export_output(video_id, transcript, "subtitle")
                    st.toast("Saved Transcript", icon="✅")

                    # Save Cleaned transcript
                    with st.spinner("Cleaning transcript..."):
                        cleaned_transcript = bot.clean_subtitles(transcript)
                        transcript_title = f"{video_id}"
                        export_output(transcript_title, cleaned_transcript, "cleaned_subtitle")
                    st.toast("Saved cleaned transcript", icon="✅")

                    with st.spinner("Embedding and Indexing video into knowledge base..."):
                        st.session_state.vector_db = index_single_video(
                            api_key=api_key,
                            video_id=video_id,
                            cleaned_transcript=cleaned_transcript
                        )
                    st.toast("Knowledge base updated successfully! 🎉", icon="✅")
                    
                    col1, col2 = st.tabs(["Summary", "Test My Knowledge"])

                    with col1:
                        with st.spinner("Generating transcript summary..."):
                            transcript_summary = bot.get_summary(transcript)
                            # Save summary
                            export_output(video_id, transcript_summary, "summary")
                            st.markdown(transcript_summary)
                        
                    with col2:
                        st.markdown("Comming Soon!")
        else:
            st.error("Invalid YouTube URL. Please check the link and try again.")

#--------------- Page 2: RAG Chatbot ---------------------------#
elif page == "RAG Chatbot":
    st.title("RAG Chatbot 🤖")

    # Check if a video has been selected first
    if not st.session_state.active_video_id:
        st.info("⚠️ Please go to the **YouTube Learning** page and load a video before using the Chatbot.")
    else:
        st.caption(f"Currently chatting about Video ID: `{st.session_state.active_video_id}`")

        # Initialize chat history
        # if "messages" not in st.session_state:
        #     st.session_state.messages = []

        if st.session_state.last_video_id != st.session_state.active_video_id:
            st.session_state.messages = []
            st.session_state.last_video_id = st.session_state.active_video_id
    
        # Display previous messages
        for message in st.session_state.messages:
            if isinstance(message, dict):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # User input
        user_input = st.chat_input("Ask something...")

        if user_input:
            # Store user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })

            with st.chat_message("user"):
                st.markdown(user_input)

            # Call RAG function
            with st.spinner("Thinking..."):
                # Pass both active_video_id and session history to the backend engine
                response, context, retrieved_ids = bot.ask_about_yt_video(
                    query=user_input,
                    vector_db=st.session_state.vector_db,
                    video_id=st.session_state.active_video_id, 
                    chat_history=st.session_state.messages[:-1]  # Exclude the query we just added
                )

            # if isinstance(response, dict):
            #     response = response.get("answer", str(response))

            # Store assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            with st.chat_message("assistant"):
                st.markdown(response)