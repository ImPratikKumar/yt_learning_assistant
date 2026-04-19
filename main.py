import os
import streamlit as st
from src.processor import extract_video_id, get_transcript
from src.llm_engine import LearningBot
from src.exporter import generate_pdf, export_output
from src.vector_store import subtile_pdf_to_db, subtile_md_to_db
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the bot once
bot = LearningBot(api_key=api_key)

# vector database
vector_db = subtile_md_to_db(api_key=api_key)

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

                    # Save transcript
                    transcript_title = f"{video_id}"
                    export_output(transcript_title, transcript, "subtitle")

                    # Save Cleaned transcript
                    cleaned_transcript = bot.clean_subtitles(transcript)
                    transcript_title = f"{video_id}"
                    export_output(transcript_title, cleaned_transcript, "cleaned_subtitle")

                    if "Error" in transcript:
                        st.error(transcript)
                    else:
                        col1, col2 = st.tabs(["Summary", "Test My Knowledge"])

                        with col1:
                            transcript_summary = bot.get_summary(transcript)

                            # Save summary
                            summary_title = f"{video_id}"
                            export_output(summary_title, transcript_summary, "summary")

                            st.markdown(transcript_summary)
                        
                        with col2:
                            st.markdown("Comming Soon!")
        else:
            st.error("Invalid YouTube URL. Please check the link and try again.")

#--------------- Page 2: RAG Chatbot ---------------------------#
elif page == "RAG Chatbot":

    st.title("RAG Chatbot 🤖")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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
            response = bot.ask_about_yt_video(user_input, vector_db)

            if isinstance(response, dict):
                response = response.get("answer", str(response))

        # Store assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        with st.chat_message("assistant"):
            st.markdown(response)