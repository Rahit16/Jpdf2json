import os
import io
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import google.generativeai as genai
from pypdf import PdfReader
from fastapi.responses import HTMLResponse

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

# Prompt to guide the Gemini AI
PROMPT = """
You are a real estate data extraction expert. I will provide you with the text content of a Japanese real estate PDF document. Your task is to extract specific real estate information and format it as a single JSON object.

The output JSON MUST have the following keys in English. The extracted values should also be translated into English. If a value is not present in the document or a direct translation is not possible (e.g., a specific address), use null.

- **Property Type:** Look for '戸建て', 'マンション', '土地', '1棟マンション', 'アパート'. Translate the found value to English (e.g., '戸建て' -> 'Detached House').
- **Price:** Look for '価格', '値段', '販売価格'.
- **Address:** Look for '所在', '所在地', '住所'.
- **Area:** Look for '面積', '土地面積', '延床面積', '延床', '敷地面積', '建物面積'.
- **Ownership:** Look for '所有権', '借地権', '敷地権'. Translate to English (e.g., '所有権' -> 'Freehold').
- **Shared Ownership:** Look for '持ち分', '共有持分'.
- **Land Category:** Look for '地目', '宅地', '山林'. Translate to English.
- **Road Info:** Look for '道路', '幅員', '長さ', '接道'.
- **Coverage & Floor-to-Area Ratio:** Look for '建ぺい率', '容積率'.
- **Zoning:** Look for '用途地域'. Translate to English.
- **Utilities:** Look for '水道', '下水', 'ガス', '都市ガス', '電気'. Translate to English.
- **Status:** Look for '現況', '居住中', '空き家', '空室'. Translate to English (e.g., '居住中' -> 'Occupied').
- **Transportation:** Look for '駅', '徒歩', '分', '沿線', '交通'.
- **Construction Date:** Look for '築年月', '建築年月', '増改築'.
- **Floor Plan & Structure:** Look for '間取り', '構造', '鉄筋コンクリート', '鉄筋鉄骨コンクリート', '鉄骨', '重量鉄骨', '軽量鉄骨', '木造'. Translate to English (e.g., '木造' -> 'Wooden').
- **Parking:** Look for '車庫', '駐車場'. Translate to English.

The output MUST be a valid JSON object. Do not include any text before or after the JSON.
"""

def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    """Extracts text from a PDF file."""
    reader = PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text



@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Returns a simple HTML landing page.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Data Extractor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f4f4f4;
                color: #333;
            }
            .container {
                max-width: 600px;
                margin: auto;
                background: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                color: #5C677D;
            }
            p {
                line-height: 1.6;
            }
            .link-btn {
                display: inline-block;
                padding: 10px 15px;
                background-color: #5C677D;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin-top: 10px;
            }
            .link-btn:hover {
                background-color: #4A5568;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to the Real Estate PDF Extractor!</h1>
            <p>This API allows you to extract key real estate data from Japanese PDF documents using the Gemini AI model.</p>
            <p>You can use the interactive API documentation to test the endpoints and understand the functionality.</p>
            <a href="/docs" class="link-btn">Go to API Documentation</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# The rest of your code for /extract-data/ and /download-json/ goes here
@app.post("/extract-data/")
async def extract_data_from_pdf(pdf_file: UploadFile = File(...)):
    """
    Accepts a PDF file, extracts text, uses the Gemini AI to parse real estate data,
    and returns a downloadable JSON file.
    """
    try:
        # Read the PDF file content
        content = await pdf_file.read()
        file_stream = io.BytesIO(content)

        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(file_stream)
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF file.")

        # Generate the payload for the Gemini AI
        full_prompt = f"{PROMPT}\n\nDocument Text:\n{extracted_text}"
        
        # Use Gemini AI to extract data
        response = model.generate_content(full_prompt)
        
        # Parse the JSON response from Gemini
        try:
            # Clean and parse the JSON string
            json_string = response.text.strip().replace('```json', '').replace('```', '')
            extracted_data = json.loads(json_string)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse JSON from Gemini's response. The AI might not have returned a valid JSON object.")

        # Return the JSON data with a download header
        return JSONResponse(
            content=extracted_data,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=extracted_data.json"}
        )

    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(status_code=500, detail=str(e))
