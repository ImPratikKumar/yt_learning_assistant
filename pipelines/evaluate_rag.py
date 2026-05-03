import os
import json
import pickle
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from datasets import Dataset
from src.llm_engine import LearningBot
from src.metrics import hit_rate_at_k, recall_at_k, precision_at_k
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

bot = LearningBot(api_key=api_key)

## Step 1: Loading the evaluation dataset
def load_eval_dataset(path="eval_dataset.json"):
    with open(path, "r") as f:
        return json.load(f)
    
## Step 2: Loading vector DB
def load_vector_db(api_key):
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=OpenAIEmbeddings(api_key=api_key)
    )

## Step 3: Retrieval + Generation
def run_rag(sample, vector_db, bot):
    question = sample["question"]

    # docs = retriever.get_relevant_documents(question)
    # contexts = [doc.page_content for doc in docs]
    # retrieved_ids = [doc.metadata["id"] for doc in docs]
    
    answer, contexts, retrieved_ids = bot.ask_about_yt_video(question, vector_db)

    return {
        "question": question,
        "contexts": contexts,
        "retrieved_ids": retrieved_ids,
        "answer": answer,
        "ground_truth": sample["ground_truth_answer"],
        "relevant_chunks": sample["relevant_chunks"]
    }

## Step 4 & 5: Running Evaluation loop using Custom Retrieval metrices
results = []
print("Started loading evaluation dataset...")
dataset = load_eval_dataset()
dataset2 = dataset[:10]
print("Started loading vector DB...")
vector_db = load_vector_db(api_key=api_key)

print("Started generating RAG response...")
for sample in tqdm(dataset2):
    output = run_rag(sample, vector_db, bot)

    hit = hit_rate_at_k(output['relevant_chunks'], output['retrieved_ids'])
    recall = recall_at_k(output['relevant_chunks'], output['retrieved_ids'])
    precision = precision_at_k(output['relevant_chunks'], output['retrieved_ids'])

    output.update({
        "hit_rate": hit,
        "recall": recall,
        "precision": precision
    })

    results.append(output)


## Step 6: Preparing data from RAGAS
# ragas_data = Dataset.from_dict({
#     "question": [r["question"] for r in results],
#     "answer": [r["answer"] for r in results],
#     "contexts": [r["question"] for r in results],
#     "ground_truth": [r["question"] for r in results]
# })

## Step 7: Running RAGAS Evaluation

# from ragas import evaluate
# from ragas.metrics import (
#     _faithfulness,
#     _answer_relevance,
#     _context_precision,
#     _context_recall
# )


# ragas_result = evaluate(
#     ragas_data,
#     metrics=[
#         _faithfulness,
#         _answer_relevance,
#         _context_precision,
#         _context_recall
#     ]
# )

## Step 8: Combining results
# final_scores = ragas_result.to_pandas().mean().to_dict()


## RAG Evaluation
question = [r["question"] for r in results]
answer = [r["answer"] for r in results]
retrieved_contexts = [[r["contexts"]] if isinstance(r["contexts"], str) else r["contexts"] for r in results]
ground_truth = [r["ground_truth"] for r in results]

ragas_dataset = []

for q, a, rc, gt in zip(question, answer, retrieved_contexts, ground_truth):
    ragas_dataset.append({
        "user_input": q,
        "retrieved_contexts": rc,
        "response": a,
        "reference": gt
    })

from ragas import EvaluationDataset
evaluation_dataset = EvaluationDataset.from_list(ragas_dataset)

import openai
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper

llm = ChatOpenAI(model="gpt-4o-mini")
openai_client = openai.OpenAI()
embedding = OpenAIEmbeddings(client=openai_client)

evaluator_llm = LangchainLLMWrapper(llm)

from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

print(f"Started RAGAS evaluation...")
ragas_result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm
)

print(f"RAGAS Result: {ragas_result}")
with open("ragas_result.pkl", "wb") as f:
    pickle.dump(ragas_result, f)

ragas_result_dict = ragas_result.to_pandas()[['context_recall', 'faithfulness', 'factual_correctness(mode=f1)']].mean().to_dict()

## Step 9: Saving everything
with open("eval_results.json", "w") as f:
    json.dump({
        "summary": ragas_result_dict,
        "detailed": results
    }, f, indent=2)