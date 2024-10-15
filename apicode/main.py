from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import pipeline

app = FastAPI()

# Load your LLM model
llm = pipeline("question-answering")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Welcome to the Q&A Application</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 20px;
                }
                h1 {
                    color: #4a4a4a;
                    text-align: center;
                }
                h2 {
                    color: #333;
                    text-align: center;
                }
                .container {
                    max-width: 600px;
                    margin: auto;
                    padding: 20px;
                    background: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                }
                p {
                    text-align: center;
                    font-size: 1.1em;
                }
                input[type="text"], input[type="submit"] {
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
                <h1>Welcome to the Q&A Application!</h1>
                <h2>Ask Your Questions</h2>
                <p>This application allows you to ask questions and get answers based on the provided context. Simply enter your context and your question below.</p>
                <form action="/ask_question/" method="post">
                    <input type="text" name="context" placeholder="Provide context for your question" required>
                    <input type="text" name="question" placeholder="Ask your question" required>
                    <input type="submit" value="Submit">
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/ask_question/")
async def ask_question(context: str = Form(...), question: str = Form(...)):
    try:
        # Analyze using the LLM model
        result = llm(question=question, context=context)

        return JSONResponse(content={"answer": result['answer']})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
