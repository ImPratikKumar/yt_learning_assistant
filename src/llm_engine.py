from openai import OpenAI
from langchain_openai import ChatOpenAI

class LearningBot:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def _ask_gpt(self, system_prompt, user_content):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        return response.choices[0].message.content
    
    def get_summary(self, transcript):
        sys = f"""
            You are an expert note-taker.

            Your job is to convert a YouTube transcript into structured, high-quality study notes.

            Instructions:
            - Extract only important ideas (ignore filler, repetition, greetins)
            - Keep it concise but informative
            - Use bullet points
            - Group related ideas under headings
            - Highlight key concepts, definitions, and examples
            - Preserve logical flow of the video

            Output format:

            Title: <generate a relevant title>

            ## Key Topics
            - Topic 1
            - Topic 2

            ## Notes
            - Main Point 1
            - Supporting detail
            - Main point 2

            ## Key Takeaways
            - Insight 1
            - Insight 2
        """
        return self._ask_gpt(sys, f"Transcript: {transcript}")
    
    def get_quiz(self, transcript):
        sys = "Create 3 hard MCQs based on this transcript."
        return self._ask_gpt(sys, transcript)
    
    def ask_about_yt_video(self, query, vector_db):
        # Search for the top relevant chunks
        docs = vector_db.similarity_search(query, k=5)
        context = "\n".join([d.page_content for d in docs])

        # Create prompt using the query and context
        prompt = f"""
            Using the following YouTube subtitle excerpts,
            Answer the following: {query}
        
            Context: {context}
        """
        model = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
        result = model.invoke(prompt).content
        return result