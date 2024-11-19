import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from fastapi import FastAPI
from pydantic import BaseModel
import lang_tutor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Virtual Tutor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; specify list for specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    '''For data validation of input paramters of chat api endpoint'''
    chat_id: str
    question: str

@app.post("/")
async def chat(request: ChatRequest):
    chat_id = request.chat_id
    question = request.question 
    response = lang_tutor.response(chat_id, question)
    return {
        "response": response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    
