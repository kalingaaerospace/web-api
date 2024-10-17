from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load the model
model = pipeline('text-generation', model='gpt2')

@app.get("/")
def read_root():
    return {"Hello": "Welcome to my AI API"}

@app.post("/analyze/")
def analyze_research(text: str):
    # Generate text or analysis using the model
    response = model(text, max_length=50)
    return {"analysis": response[0]['generated_text']}
