from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
import pdfplumber

app = FastAPI()

# Load your LLM model
llm = pipeline("question-answering")

# Mount static files for HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>File Upload and Q&A</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 20px;
                }
                h2 {
                    color: #333;
                }
                .container {
                    max-width: 600px;
                    margin: auto;
                    padding: 20px;
                    background: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                }
                input[type="file"], input[type="text"], input[type="submit"] {
                    width: 100%;
                    padding: 10px;
                    margin: 10px 0;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                input[type="submit"] {
                    background-color: #5cb85c;
                    color: white;
                    border: none;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #4cae4c;
                }
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
        contents = await file.read()
        
        # Use pdfplumber to extract text from PDF
        text = ""
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"

        # Analyze using the LLM model
        result = llm(question=question, context=text)

        return JSONResponse(content={"answer": result['answer']})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
