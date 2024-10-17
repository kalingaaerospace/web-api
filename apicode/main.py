from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load the model
model = pipeline('text-generation', model='gpt2')

@app.get("/")
def read_root():
    return {"Hello": "Welcome to my AI API"}
