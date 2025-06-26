import fitz
import os
import numpy as np
import json
from dotenv import load_dotenv
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


load_dotenv()

def extract_text_from_pdf(pdf_path):

    mypdf = fitz.open(pdf_path)
    all_text = ""

    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text


def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        content = file.read()

    return content

def chunk_text(text, n, overlap):

    chunks = []
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]
        chunks.append([chunk])

    return chunks

def create_embeddings(text):

    api_key=os.environ.get("ARK_API_KEY")
    url="https://api.siliconflow.cn/v1/embeddings"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": text
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    return response.json()["data"][0]["embedding"]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=5):
    query_embedding = create_embeddings(query)
    similarity_scores = []

    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding))
        similarity_scores.append((i, similarity_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]

    return [text_chunks[index] for index in top_indices]

def generate_response(system_prompt, user_message):

    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "messages": [

            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}

        ]
    }

    headers ={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('ARK_API_KEY')}",
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    content = ""
    result = json.loads(response.text).get('choices', [])
    for r in result:
        content = r.get('message', {}).get('content', '')

    return content

global_text_chunks = None
global_embeddings = None

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question
    if global_text_chunks is None or global_embeddings is None:
        return {"answer": "错误: 没有加载文档，请先启动服务"}

    top_chunks = semantic_search(question, global_text_chunks, global_embeddings, 10)

    system_prompt = "你是一名 AI 助手，能够严格按照给定的上下文生成回答。如果无法直接根据给定的上下文生成回答，仅输出'抱歉，RAG知识库中暂无相关信息'"

    user_prompt = "\n".join(
        [f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
    user_prompt = f"{user_prompt}\nQuestion: {question}"

    # 生成回答
    ai_response = generate_response(system_prompt, user_prompt)

    return {"answer": ai_response}




if __name__ == "__main__":
    #pdf_path = "data/test_document.pdf"
    #extracted_text = extract_text_from_pdf(pdf_path)
    extracted_text = extract_text_from_txt(r"C:\Users\LENOVO\Desktop\zhuanyeshixun\qt\login\pybackend\simpleRAG\data\test.txt")
    text_chunks = chunk_text(extracted_text, 200, 50)

    response = []
    for i in range(len(text_chunks)):
        response.append(create_embeddings(text_chunks[i]))

    global_text_chunks = text_chunks
    global_embeddings = response

    print(f"共生成 {len(global_text_chunks)} 个文本块")
    print("RAG系统启动成功")

    uvicorn.run(app, host="0.0.0.0", port=8000)