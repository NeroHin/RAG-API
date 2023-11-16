from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uvicorn
import logging

# 初始化 FastAPI 應用
app = FastAPI()

# 讀取 CSV 數據
model = SentenceTransformer("BAAI/bge-large-zh", cache_folder="../RAG-API/cache_models/")
qa_df = pd.read_csv("loan_data.csv")
question_embeddings = model.encode(qa_df["input"].values.tolist())


logging.basicConfig(level=logging.INFO)
# print embeddings finished
logging.info("Embeddings finished")

# 定義請求體
class Query(BaseModel):
    question: str

# API 端點
@app.post("/get_similar_questions")
async def get_similar_questions(query: Query):
    question_embedding = model.encode([query.question])
    
    print(query.question)
    
    print(type([query.question]))

    # 計算相似度
    # similarities = cosine_similarity(question_embedding, question_embeddings)[0]

    # # 從相似度數組中找出最相似的三個問題的索引
    # top_3_indices = np.argsort(similarities)[-3:][::-1]

    # if similarities[top_3_indices[0]] < 0.5:
    #     raise HTTPException(status_code=404, detail="Question not found")
    
    # # 獲取相關的問題和答案
    # responses = []
    # for index in top_3_indices:
    #     responses.append({
    #         'question': qa_df.iloc[index]["input"],
    #         'answer': qa_df.iloc[index]["output"],
    #         'similarity': similarities[index]
    #     })

    return {"similar_questions":question_embedding}

# 啟動服務器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
