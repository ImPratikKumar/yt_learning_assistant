import asyncio
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
    
    # def get_summary(self, transcript):
        
    #     model = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
    #     splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=6000,
    #         chunk_overlap=800,
    #         separators=["\n\n", "\n", ".", " "]
    #     )

    #     # Step 1: Chunking
    #     chunks = splitter.split_text(transcript)

    #     # Step 2: Chunk notes
    #     partial_notes = []
    #     for chunk in chunks:
    #         chunk_prompt = f"""
    #             You are an expert note-taker converting a YouTube lecture transcript into structured study notes.
            
    #             Context:
    #             This is a PART of a longer video. Maintain continuity and avoid assuming this is the full content.
            
    #             Instructions:
    #             - Extract only important ideas (ignore filler, repetition, greetings)
    #             - Write concise but informative notes
    #             - Use bullet points (no long paragraphs)
    #             - Preserve logical flow with this chunk
    #             - Capture:
    #                 - Key concepts
    #                 - Definitions
    #                 - Examples (if any)
    #                 - Steps / processes (if explained)
    #             - Avoid repeating the same idea multiple times
            
    #             Output Format:
            
    #             ## Topic Covered
    #             - <topic 1>
    #             - <topic 2>
            
    #             ## Key Points
    #             - <main idea 1>
    #                 - <supporting detail>
    #             - <main idea 2>
            
    #             ## Important Terms (if any)
    #             - Term: Definition
            
    #             ## Examples (if any)
    #             - <example>
            
    #             Transcript Chunk:
    #             {chunk}
    #         """

    #         result = model.invoke(chunk_prompt)
    #         partial_notes.append(result.content)

    #     partial_notes_joined = "\n\n".join(partial_notes)

    #     # Step 3: Merge
    #     merge_prompt = f"""
    #         You are an expert editor and educator.

    #         Below are structured notes extracted from different parts of a YouTube video.

    #         Your job:
    #         - Merge them into a SINGLE, coherent, well-structured note
    #         - Restore logical flow of the original video
    #         - Remove redundanc and repeated points
    #         - Group related ideas under meaningful sections
    #         - Ensure smooth transitions between sections
    #         - Keep it concise but complete

    #         DO NOT:
    #         - Repeat the sampe concept multiple times
    #         - Keep chunk boundaries visible
    #         - Include irrelevant or low-value information

    #         Output Format:

    #         Title: <Generate a clear and revelant title>

    #         ## Detailed Notes

    #         ### <Section 1>
    #         - Key idea
    #             - Supporting detail

    #         ### <Section 2>
    #         - Key idea

    #         ## Key Takeaways
    #         - Most important insights from the video

    #         Notes to Merge:
    #         {partial_notes_joined}
    #     """
    #     final_notes = model.invoke(merge_prompt).content

    #     # Step 4: Refine
    #     refine_prompt = f"""
    #         You are a senior education refining study notes.

    #         Improve the following notes by:
    #         - Making them clearer and more structured
    #         - Improving readability
    #         - Ensuring logical flow
    #         - Removing any remaining redundancy
    #         - Keeping it concise but insightful

    #         Enhancements:
    #         - Add better section heading if needed
    #         - Simplify complex explanations
    #         - Ensure formatting is consistent

    #         Final Notes:
    #         {final_notes}
    #     """
    #     refined_notes = model.invoke(refine_prompt).content

    #     return refined_notes

    async def _async_generate_chunk_note(self,model, chunk, index, total):
        """Helper async function to process a single chunk using .ainvoke()"""
        
        chunk_prompt = f"""
            You are an expert note-taker converting a YouTube lecture transcript into structured study notes.
        
            Context:
            This is PART {index+1} of a longer video containing {total} parts. Maintain continuity and aviod assuming this is the full content.
        
            Instructions:
            - Extract only import ideas (ignore filler, repetition, greetings)
            - Write concise but informative notes
            - Use bullet points (no long paragraphs)
            - Preserve logical flow with this chunk
            - Capture:
                - Key concepts
                - Definitions
                - Expamples (if any)
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

        result = await model.ainvoke(chunk_prompt)
        return result.content
    
    async def get_summary_async(self, transcript):
        """Asynchronously chunks, summarizes, merges, adn refines the transcript."""
        
        model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=6000,
            chunk_overlap=800,
            separators=["\n\n", "\n", ".", " "]
        )
        
        # Step 1: Chunking
        chunks = splitter.split_text(transcript)
        total_chunks = len(chunks)
        
        # Step 2: Concurrent Chunk Notes Generation
        # Create coroutine tasks for all chunks to run concurrently
        tasks = [
            self._async_generate_chunk_note(model, chunk, i, total_chunks)
            for i, chunk in enumerate(chunks)
        ]
        
        # Fire off all API calls concurrently and wait for all to finish
        partial_notes = await asyncio.gather(*tasks)
        
        partial_notes_joined = "\n\n".join(partial_notes)
        
        
        # Step 3: Merge
        merge_prompt = f"""
            You are an expert editor and educator.
        
            Below are structured notes extracted from different part of a YouTube video.
        
            Your job:
            - Merge them into a SINGLE, coherent, well-structured note
            - Restore logical flow of the origial video
            - Remove redundancy and repeated points
            - Group related ideas under meaningful sections
            - Ensure smooth transitions between sections
            - Keep it concise but complete
        
            DO NOT:
            - Repeat the sampe concept multiple times
            - Keep chunk boundaries visible
            - Include irrelevant or low-value information
        
            Output Format:
        
            Title: <Generate a clear and relevant title>
        
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
        
        final_notes_res = await model.ainvoke(merge_prompt)
        final_notes = final_notes_res.content
        
        # Step 4: Refine
        refine_prompt = f"""
            You are a senior educatior refining study notes.
        
            Improve the following notes by:
            - Making them clearer and more strucured
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
        
        refined_notes_res = await model.ainvoke(refine_prompt)
        refined_notes = refined_notes_res.content
        
        return refined_notes
    
    def get_summary(self, transcript):
        return asyncio.run(self.get_summary_async(transcript))
    
    def get_quiz(self, transcript):
        sys = "Create 3 hard MCQs based on this transcript."
        return self._ask_gpt(sys, transcript)
    
    # def ask_about_yt_video(self, query, vector_db):
    #     # Search for the top relevant chunks
    #     docs = vector_db.similarity_search(query, k=5)
    #     context = "\n".join([d.page_content for d in docs])
    #     retrieved_ids = [doc.metadata["id"] for doc in docs]

    #     # Create prompt using the query and context
    #     prompt = f"""
    #         Using the following YouTube subtitle excerpts,
    #         Answer the following: {query}
        
    #         Context: {context}
    #     """
    #     model = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
    #     result = model.invoke(prompt).content
    #     return result, context, retrieved_ids

    def ask_about_yt_video(self, query, vector_db, video_id, chat_history=[]):
        """
        Answers user questions by first rephrasing follow-up inputs using chat history,
        and then searching ChromaDB with a strict metadata filter for the active video.
        """
        model = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)

        # 1. Contextualize / Rephrase the query if chat history exists
        standalone_query = query
        if chat_history:
            # Format history for the contextualizer prompt
            history_str = ""
            for msg in chat_history[-5:]: # Look at the last 5 exchanges for speed/tloen economy
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n"

            rephrase_prompt = f"""
                Given the following chat history and a follow-up question,rephrase the follow-up
                question into a standalone question that can be understood WITHOUT the chat history.
                If it is already a standalone question, return it exactly as it is.
                DO NOT answer the question, just rephrase it.

                Chat History:
                {history_str}

                Follow-up Question: {query}

                Standalone Question:
            """
            try:
                standalone_query = model.invoke(rephrase_prompt).content.strip()
            except Exception:
                standalone_query = query
            # 2. Search ChromaDB with a strict metadata filter for this video
            # This prevents chunks from other videos from leaking into your context!
        try:
            docs = vector_db.similarity_search(
                standalone_query,
                k=5,
                filter={"video_id": video_id}
            ) 
        except Exception:
            # Fallback if the database structure hasn't been re-initialized with the filter keys
            docs = vector_db.similarity_search(standalone_query, k=5)

        context = "\n".join([d.page_content for d in docs])
        retrieved_ids = [d.metadata.get("id", "unkown") for d in docs]

        # 3. Generate final answer with localized context
        prompt = f"""
            You are a helpful educational learning assistant. Using the following YouTube subtitle excerpts,
            answer the following question comprehensively. If the context doesn't contain the answer, use your general
            knowledge but explicitly clarify that it wasn't mentioned in the video.

            Context from video:
            {context}

            Question: {standalone_query}

            Answer:
        """
        result = model.invoke(prompt).content
        return result, context, retrieved_ids

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