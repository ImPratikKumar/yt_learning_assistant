import re
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from various YouTube URL formats.
    Supported:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/v/VIDEO_ID
    """
    # Regex pattern for most common YouTube URL structures
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'

    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

def get_transcript(video_id: str) -> str:
    """
    Fetches the transcript and joins it into a single clean string.
    """
    try:
        # transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = YouTubeTranscriptApi().fetch(video_id)
        # Join the text snippets with a space
        # full_text = " ".join([entry['text'] for entry in transcript_list])
        full_text = " ".join([t['text'] for t in transcript.to_raw_data()])
        return full_text
    except Exception as e:
        # Common error: Subtitles disabled, Video not found, or Request blocked
        return f"Error fetching transcript: {str(e)}"

# def get_transcript(video_id):
#     """
#     Returns the subtitle for given video id.
#     input: video_id
#     output: subtitle

#     function input example:
#     youtube_video_url = "https://www.youtube.com/watch?v=FSh8eusFYL4&t=473s"
#     video_id = "FSh8eusFYL4"
#     """
#     from youtube_transcript_api import YouTubeTranscriptApi
#     try:
#         transcript = YouTubeTranscriptApi().fetch(video_id)
#         # subtitle = ""
#         # for i in range(len(transcript.to_raw_data())):
#         #     subtitle += transcript.to_raw_data()[i]['text']
#         subtitle = " ".join([t['text'] for t in transcript.to_raw_data()])
#         return subtitle
#     except Exception as e:
#         return f"Error: Could not retrive transcript. (Check if captions are disabled): {e}"