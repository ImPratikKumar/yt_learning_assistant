from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        
        model = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=6000,
            chunk_overlap=800,
            separators=["\n\n", "\n", ".", " "]
        )

        # Step 1: Chunking
        chunks = splitter.split_text(transcript)

        # Step 2: Chunk notes
        partial_notes = []
        for chunk in chunks:
            chunk_prompt = f"""
                You are an expert note-taker converting a YouTube lecture transcript into structured study notes.
            
                Context:
                This is a PART of a longer video. Maintain continuity and avoid assuming this is the full content.
            
                Instructions:
                - Extract only important ideas (ignore filler, repetition, greetings)
                - Write concise but informative notes
                - Use bullet points (no long paragraphs)
                - Preserve logical flow with this chunk
                - Capture:
                    - Key concepts
                    - Definitions
                    - Examples (if any)
                    - Steps / processes (if explained)
                - Avoid repeating the same idea multiple times
            
                Output Format:
            
                ## Topic Covered
                - <topic 1>
                - <topic 2>
            
                ## Key Points
                - <main idea 1>
                    - <supporting detail>
                - <main idea 2>
            
                ## Important Terms (if any)
                - Term: Definition
            
                ## Examples (if any)
                - <example>
            
                Transcript Chunk:
                {chunk}
            """

            result = model.invoke(chunk_prompt)
            partial_notes.append(result.content)

        partial_notes_joined = "\n\n".join(partial_notes)

        # Step 3: Merge
        merge_prompt = f"""
            You are an expert editor and educator.

            Below are structured notes extracted from different parts of a YouTube video.

            Your job:
            - Merge them into a SINGLE, coherent, well-structured note
            - Restore logical flow of the original video
            - Remove redundanc and repeated points
            - Group related ideas under meaningful sections
            - Ensure smooth transitions between sections
            - Keep it concise but complete

            DO NOT:
            - Repeat the sampe concept multiple times
            - Keep chunk boundaries visible
            - Include irrelevant or low-value information

            Output Format:

            Title: <Generate a clear and revelant title>

            ## Detailed Notes

            ### <Section 1>
            - Key idea
                - Supporting detail

            ### <Section 2>
            - Key idea

            ## Key Takeaways
            - Most important insights from the video

            Notes to Merge:
            {partial_notes_joined}
        """
        final_notes = model.invoke(merge_prompt).content

        # Step 4: Refine
        refine_prompt = f"""
            You are a senior education refining study notes.

            Improve the following notes by:
            - Making them clearer and more structured
            - Improving readability
            - Ensuring logical flow
            - Removing any remaining redundancy
            - Keeping it concise but insightful

            Enhancements:
            - Add better section heading if needed
            - Simplify complex explanations
            - Ensure formatting is consistent

            Final Notes:
            {final_notes}
        """
        refined_notes = model.invoke(refine_prompt).content

        return refined_notes
    
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

    def clean_subtitles(self, text):

        prompt = f"""
            Convert the following YouTube subtitles into clean, well-structured text:
            - Fix punctuation
            - Merge broken sentences
            - Keep meaning intact

            Text:
            {text} 
        """

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        result = model.invoke(prompt).content
        return result