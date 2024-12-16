
from dotenv import load_dotenv
import google.generativeai as genai
import os

from utils import parse_llm_json

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro")

def gemini_parse_image(img, prompt):
    response = gemini_model.generate_content([img, prompt])
    try:
        res = parse_llm_json(response.text)
    except:
        res = {'score': -1}
        print("WARNING: Failed to parse response from Gemini, got:", response.text)
    return res
