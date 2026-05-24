import os
import json

HISTORY_FILE = "./data/video_history.json"

def load_video_history() -> dict:
    """Load the dictionary of previously processed videos."""
    if not os.path.exists(HISTORY_FILE):
        # Create dictory if it doesn't exist
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
    
def save_video_to_history(video_id: str, title: str):
    """Save a video IS and its friendly title/name to the catalog map."""
    history = load_video_history()
    # Using video_id as a key natually prevents duplicates in our tracking index
    history[video_id] = {
        "video_id": video_id,
        "title": title
    }

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

def is_video_processed(video_id: str) -> bool:
    """Checks if a video has already been indexed"""
    history = load_video_history()
    return video_id in history