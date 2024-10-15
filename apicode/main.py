from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
import pdfplumber
import os

app = FastAPI()

# Load a lightweight model
llm = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>File Upload and Q&A</title>
            <style>
                body { font-family: Arial, sans-serif; background-color: #f4f4f4; }
                h2 { color: #333; }
                .container { max-width: 600px; margin: auto; padding: 20px; background: white; }
                input[type="file"], input[type="text"], input[type="submit"] { width: 100%; padding: 10px; }
                input[type="submit"] { background-color: #5cb85c; color: white; border: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Upload File and Ask a Question</h2>
                <form action="/uploadfile/" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" required>
                    <input type="text" name="question" placeholder="Ask your question" required>
                    <input type="submit" value="Submit">
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), question: str = Form(...)):
    try:
        text = ""
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"

        result = llm(question=question, context=text)
        return JSONResponse(content={"answer": result['answer']})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

