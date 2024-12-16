from typing import Dict
import json, os, time, shutil

def parse_llm_json(llm_output: str) -> Dict:
    if llm_output[:3] == "```":
        llm_output = llm_output.strip("`\n ")
    if llm_output[:4] == "json":
        llm_output = llm_output[5:]
    return json.loads(llm_output)

def list_folders(directory):
    return [name for name in os.listdir(directory)]

def countdown_with_message(t, msg):
    for i in range(t):
        print(f"{msg} in {t - i} seconds...", end='\r')
        time.sleep(1)
    return

def delete_entire_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
