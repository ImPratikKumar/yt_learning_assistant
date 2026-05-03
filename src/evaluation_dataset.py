import os
import json
import uuid
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

## Generate High-quality Q&A
def generate_qa(chunk_text):

    prompt = f"""
        You are creating a evaluation data for a learning system.

        From the following content, generate 3 high-quality Q&A pairs.

        Rules:
        - Question must be specific and answerable from the text.
        - Avoid trivial or surface-level questions.
        - Include a mix of:
            - factual
            - conceptual (why/how)
            - applied (use-case or implication)
        - Answers must be precise and fully grounded in the text
        - DO NOT copy sentences directly; rephrase

        Return JSON format:
        [
            {{
                "question": "...".
                "answer": "...",
                "type": "factual/conceptual/applied",
                "difficulty": "easy/medium/hard"
            }}
        ]

        Return ONLY valid JSON. No explanation.

        Content:
        {chunk_text}
    """

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    try:
        result = model.invoke(prompt).content
        return json.loads(result)
    except:
        return []
    
def filter_qa(qa, context):

    prompt = f"""
        Question: {qa['question']}
        Answer: {qa['answer']}

        Context:
        {context}

        Check:
        1. Is the question meaningful and non-trivial?
        2. Is the answer fully supported by the context?
        3. Is this useful for learning?

        Return:
        PASS or FAIL

        Return ONLY PASS or FAIL. No Explanation
    """

    model = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
    result = model.invoke(prompt).content.strip().upper()
    return 'PASS' in result

def generate_multi_chunk_qa(chunks):

    dataset = []

    for i in tqdm(range(0, len(chunks)-1, 2)):
        combined = chunks[i].page_content + "\n" + chunks[i+1].page_content

        prompt = f"""
            Create a complex question requiring combining ideas.

            Content:
            {combined}

            Return JSON:
            {{
                "question": "...",
                "answer": "...",
                "type": "conceptual",
                "difficulty": "hard"
            }}

            Return ONLY valid JSON. No Explanation
        """

        model = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
        try:
            qa = json.loads(model.invoke(prompt).content)
            qa["relevant_chunk"] = [chunks[i].metadata["id"], chunks[i+1].metadata["id"]]
            dataset.append(qa)
        except:
            continue

    return dataset

def build_dataset(chunks):

    dataset = []

    print("Started generating Q&A on single chunk...")
    for chunk in tqdm(chunks):
        qa_pairs = generate_qa(chunk.page_content)

        for qa in qa_pairs:
            if filter_qa(qa, chunk.page_content):
                dataset.append({
                    "id": str(uuid.uuid4()),
                    "question": qa["question"],
                    "ground_truth_answer": qa["answer"],
                    "relevant_chunks": [chunk.metadata['id']],
                    "type": qa.get("type", "conceptual"),
                    "difficulty": qa.get("difficulty", "medium")
                })

    print("Started generating Q&A on multiple chunk...")
    multi_chunk_qa = generate_multi_chunk_qa(chunks)

    for qa in multi_chunk_qa:
        dataset.append({
            "id": str(uuid.uuid4()),
            "question": qa["question"],
            "ground_truth_answer": qa["answer"],
            "relevant_chunks": qa["relevant_chunk"],
            "type": qa["type"],
            "difficulty": qa["difficulty"]
        })
        
    return dataset