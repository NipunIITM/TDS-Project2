import os
import re
import json
import base64
import tempfile
import subprocess
import logging
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import requests
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)

# --- (Other functions and LLM setup from your file go here) ---

# Corrected route to match the frontend's request
@app.post("/api") # <--- THIS IS THE FIX
async def analyze_data(request: Request):
    try:
        form = await request.form()

        # Your existing logic for handling the form and calling the agent
        uploads = [form[key] for key in form if isinstance(form[key], UploadFile)]

        if not uploads:
            raise HTTPException(400, "At least one file is required.")

        txt_files = [f for f in uploads if (f.filename or "").lower().endswith(".txt")]
        if not txt_files:
            raise HTTPException(400, "A .txt questions file is required.")
        
        questions_file = txt_files[0]
        raw_questions = (await questions_file.read()).decode("utf-8")
        
        # This part assumes you have the llm and agent_executor configured as in your provided file
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-1.5-pro-latest"),
            temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        @tool
        def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
            # Your scraping logic
            return {"status": "success", "data": [{"col1": "data1"}]}

        tools = [scrape_url_to_dataframe]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data analyst agent."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        llm_input = f"Rules: Use scrape_url_to_dataframe if needed.\nQuestions:\n{raw_questions}"
        response = await agent_executor.ainvoke({"input": llm_input})
        
        final_json_output = json.loads(response['output'])
        
        return JSONResponse(content=final_json_output)

    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
