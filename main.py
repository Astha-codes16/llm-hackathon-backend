
    
# from fastapi import FastAPI,HTTPException
# from pydantic import BaseModel
# app=FastAPI()
# items={}
# class Model(BaseModel):
#     name:str
#     description:str
# @app.post("/items/{id}")
# def create_item(id:str,item:Model):
#     items[id]=item
#     return{"message":"Item created","item":items[id]}
# @app.get("/items/{id}")
# def get_get(id:str):
#     if id in items:
#         return items[id]
#     else:
#         raise HTTPException(status_code=404,detail="item not found")
from fastapi import FastAPI, UploadFile,File,HTTPException,Depends,status,Security

from pydantic import BaseModel
from fastapi import Header
from typing import List
import os
import shutil

import openai
import fitz 
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()


Api_key="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"
vectorStore = None #global variable
# Folder to store uploaded documents
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
@app.get("/")
def read_root():
    return {"message": "FastAPI app is live!"}
class HackerxRequest(BaseModel):
    questions:List[str]
    document:str

    


def get_llm_response(prompt):
    headers = {
        "Authorization": f"Bearer {Api_key}",  
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "top_p": 0.7,
        "max_tokens": 256
    }
    
    if not Api_key:
        raise RuntimeError("Api key not found")
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()




    # from here
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Each chunk ~500 characters
        chunk_overlap=100      # 100 chars repeated between chunks (for context)
    )
    chunks = text_splitter.split_text(text)
    return chunks
   
@app.post("/hackrx/run")
async def hackrx_run(request: HackerxRequest):

    try:
        # Download the PDF from the link
        pdf_url=request.document
        response = requests.get(pdf_url)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF from the link.")

        # Save to local file
        file_path = os.path.join("uploaded_files", "linked_document.pdf")
        with open(file_path, "wb") as f:
            f.write(response.content)

        # Extract, chunk, and embed
        extracted_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(extracted_text)
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorStore = FAISS.from_texts(chunks, embedding_model)
        answers=[]
        for question in request.questions:

            docs = vectorStore.similarity_search(question, k=4)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            answer = get_llm_response(prompt)
            answers.append(answer)
        return {
            "answers":answers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

 # PyMuPDF












    