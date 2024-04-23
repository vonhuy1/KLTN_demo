from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import server_function as sf

NGROK_STATIC_DOMAIN = "rightly-poetic-amoeba.ngrok-free.app"
NGROK_TOKEN="2fEEmIGqKqV2EoE6n5lBRgObtov_4KmEfqwPJLdMi2G7GKnXU"
app = FastAPI(
    title="Python ChatGPT plugin",
    description="A Python ChatGPT plugin"
)

ALLOWED_EXTENSIONS = {'csv', 'txt', 'doc', 'docx', 'pdf', 'xlsx', 'pptx','xml','csv','sql','html','md','json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.get("/query/")
async def handle_query(question: str):
    text_all = sf.extract_data()
    test = sf.question_answer_all_query_v1(question,text_all)
    return {"message": f"{test}"}

@app.get("/extract_file/")
async def extract_file():
    text_all = sf.extract_data()
    return {"message": f"Xử lý thành công{text_all}"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File không được hỗ trợ")
    # Tạo một thư mục tạm để lưu file
    os.makedirs("temp", exist_ok=True)
    with open(f"temp/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return JSONResponse(content={"message": "File đã được tải lên và lưu thành công"}, status_code=200)
import nest_asyncio
from uvicorn import run
nest_asyncio.apply() 
# Cần áp dụng nest_asyncio trong Jupyter Notebook

if __name__ == "__main__":
    run(app, port=8000)
