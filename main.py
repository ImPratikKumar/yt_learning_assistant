import os
import streamlit as st
from src.processor import extract_video_id, get_transcript
from src.llm_engine import LearningBot
from src.exporter import generate_pdf
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initialize the bot once
bot = LearningBot(api_key=api_key)

st.title("AI Study Buddy")
video_url = st.text_input("Paste YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")


if video_url:
    video_id = extract_video_id(video_url)

    if video_id:
        st.success(f"Found Video ID: {video_id}")

        if st.button("Generate Learning Materials"):
            with st.spinner("Reading transcript..."):
                transcript = get_transcript(video_id)

                ## save the transcript
                transcript_title = f"{video_id}"
                generate_pdf(transcript_title, transcript, "subtitle")

                if "Error" in transcript:
                    st.error(transcript)
                else:
                    col1, col2 = st.tabs(["Summary", "Test My Knowledge"])
                    with col1:
                        transcript_summary = bot.get_summary(transcript)
                        ## Save the transcript summary
                        sumamry_title = f"{video_id}"
                        generate_pdf(sumamry_title, transcript_summary, "summary")
                        st.markdown(transcript_summary)

                        # # Generating the PDF file in memory
                        # pdf_data = generate_pdf(f"YT {video_id} Notes", transcript_summary)

                        # # download button
                        # st.download_button(
                        #     label="Download Notes as PDF",
                        #     data=generate_pdf(video_id, transcript),
                        #     file_name=f"{video_id}.pdf",
                        #     mime="application/pdf"
                        # )

                    with col2:
                        # st.markdown(bot.get_quiz(transcript))
                        st.markdown("Comming Soon!")
        else:
            st.error("Invalid YouTube URL. Please check the link and try again.")

# if st.button("Analyze"):
#     transcript = get_transcript(video_url)

#     col1, col2 = st.tabs(["Summary", "Test My Knowledge"])

#     with col1:
#         st.write(bot.get_summary(transcript))
#     with col2:
#         st.write(bot.get_quiz(transcript))