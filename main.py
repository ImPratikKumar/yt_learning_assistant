import os
import streamlit as st
from src.processor import extract_video_id, get_transcript
from src.llm_engine import LearningBot
from src.vector_store import index_single_video
from src.exporter import generate_pdf, export_output
from src.history_manager import load_video_history, save_video_to_history, is_video_processed
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the bot once
bot = LearningBot(api_key=api_key)

# Initializing session state variable
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

st.sidebar.markdown("---")
st.sidebar.title("Your Video Library📚")

# Load existing video index catalog map
history = load_video_history()

if history:
    # Foramt options for the select box dropdown list display
    options = {f"{v['title']} ({v['video_id']})": v['video_id'] for v in history.values()}

    # Pre-select the actuve video index if one is set in the session
    current_index=0
    if st.session_state.active_video_id:
        for idx, vid in enumerate(options.values()):
            if vid == st.session_state.active_video_id:
                current_index = idx
    
    selected_video_label = st.sidebar.selectbox(
        "select an indexed video to study:",
        options=list(options.keys()),
        index=current_index
    )
    # Automatically switch global active scope whenever selection changes
    st.session_state.active_video_id = options[selected_video_label]
else:
    st.sidebar.info("No video processed yet! Paste a link to get started.")

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
            # Set active video session ID
            st.session_state.active_video_id = video_id

            # check fpr duplications before processing
            if is_video_processed(video_id):
                st.info("💡This video has already been processed and embedded! You can jump straight to the RAG Chatbot tab to query it, or read the stored notes below.")

                # Dynamic load paths if user want to look at notes again
                col1, col2 = st.tabs(["Summary", "Test My Knowledge"])
                with col1:
                    summary_path = f"./data/summary_{video_id}.md"
                    if os.path.exists(summary_path):
                        with open(summary_path, "r", encoding="utf-8") as f:
                            st.markdown(f.read())
                    else:
                        st.warning("Summary file not found locally, rerun extraction if needed.")
                with col2:
                    st.markdown("Comming Soon!")
            else:
                # If it's a completely new video, open processing swithces
                if st.button("Generate Learning Materials"):
                    with st.spinner("Reading transcript..."):
                        transcript = get_transcript(video_id)

                    if "Error" in transcript:
                        st.error(transcript)
                    else:
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

                        # Save ti persisten catalog map
                        # Using video_id as title fallback until metadata parser is linked
                        save_video_to_history(video_id, title=f"Video {video_id}")
                    
                        col1, col2 = st.tabs(["Summary", "Test My Knowledge"])

                        with col1:
                            with st.spinner("Generating transcript summary..."):
                                transcript_summary = bot.get_summary(transcript)
                                # Save summary
                                export_output(video_id, transcript_summary, "summary")
                                st.markdown(transcript_summary)
                        
                        with col2:
                            st.markdown("Comming Soon!")
                        st.rerun() # Refresh layout so dropdown sidebar updates immediately
        else:
            st.error("Invalid YouTube URL. Please check the link and try again.")

#--------------- Page 2: RAG Chatbot ---------------------------#
elif page == "RAG Chatbot":
    st.title("RAG Chatbot 🤖")

    # Check if a video has been selected first
    if not st.session_state.active_video_id:
        st.info("⚠️ Please go to the **YouTube Learning** page and load a video before using the Chatbot.")
    else:
        # Retrieve the friendly name text configuration from history catalog map
        current_video_title = history.get(st.session_state.active_video_id, {}).get("title", st.session_state.active_video_id)
        st.caption(f"Currently chatting about: **{current_video_title}**  (`{st.session_state.active_video_id}`)")

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