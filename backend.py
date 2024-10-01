from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from genAgent import agent


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/getResponse")
async def getResponse(userInput:str):
    try:
        chatResponse = agent.main(userInput)
        print(chatResponse)
        return {"response": chatResponse}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))



# Sample API Call - To represent your backend calls
@app.get("/api/getAccountDetails")
async def getAccountDetails(accountNumber):
    try:
        if accountNumber == "123":
            accoutName = "Akhil Tadiparthi"
            return {"accountName": accoutName}
        else:
            return {"accountName": "Unknown User"}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))