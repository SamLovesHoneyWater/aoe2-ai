import cv2
import torch
#import pytesseract
from PIL import Image
import mss, time, os, re, random
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import google.generativeai as genai

from utils import parse_llm_json
from actions import *
from GameAI import GameAI

#pytesseract.pytesseract.tesseract_cmd = r'D:/Tesseract/tesseract.exe'


GAMESCREEN_X0, GAMESCREEN_Y0 = 0, 75
GAMESCREEN_X1, GAMESCREEN_Y1 = 1920, 1024

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro")

image_path = 'sample_images/continue_button_test.png'
label_img = Image.open(image_path)
label_img_np = np.array(label_img)

def gemini_parse_image(img, prompt):
    response = gemini_model.generate_content([img, prompt])
    res = parse_llm_json(response.text)
    return res

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_model = GameAI().to(device)

def get_screen(x0, y0, x1, y1):
    with mss.mss() as sct:
        monitor = {"top": y0, "left": x0, "width": x1 - x0, "height": y1 - y0}
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    return img

def is_killed():
    x0, y0 = 940, 725
    x1, y1 = 1103, 775
    img = get_screen(x0, y0, x1, y1)
    img_np = np.array(img)
    diff = np.abs(label_img_np - img_np)
    avg_diff = np.mean(diff)
    return avg_diff

def respawn():
    x0, y0 = 940, 725
    x1, y1 = 1103, 775
    x, y = (x0 + x1) // 2, (y0 + y1) // 2
    xx, yy = 960, 925
    pyautogui.moveTo(x, y)
    pyautogui.click()
    print("Respawning...")
    time.sleep(1)
    pyautogui.moveTo(xx, yy)
    pyautogui.click()
    return

def get_ingame_score():
    prompt = "Parse the text in the image and output in the following json format:\n{\n\tscore: xxx\n}"
    x0, y0 = 831, 957
    x1, y1 = 1100, 985
    img = get_screen(x0, y0, x1, y1)
    stats = gemini_parse_image(img, prompt)
    return stats

def get_final_score():
    prompt = "Parse the text in the image and output in the following json format:\n{\n\tscore: xxx,\n\tlevel: xxx,\n\ttime: xxx\n}"
    x0, y0 = 634, 494
    x1, y1 = 865, 671
    img = get_screen(x0, y0, x1, y1)
    stats = gemini_parse_image(img, prompt)
    return stats

def get_game_scene():
    img = get_screen(GAMESCREEN_X0, GAMESCREEN_Y0, GAMESCREEN_X1, GAMESCREEN_Y1)
    #img = img.convert('L')
    k = .2
    img = img.resize((int((GAMESCREEN_X1 - GAMESCREEN_X0) * k), int((GAMESCREEN_Y1 - GAMESCREEN_Y0) * k)), Image.Resampling.LANCZOS)
    print(img.size)
    return img

def get_action_ai(img):
    img = np.array(img)
    img = img.reshape(1, 3, img.shape[0], img.shape[1])
    print(img.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = torch.from_numpy(img).float().to(device)
    outputs = policy_model(img_tensor)
    actions = output_to_actions(outputs)
    return actions

def release_all_keys():
    for key in ALL_KEYS:
        keyboard.release(key)
    return

def get_human_action():
    # Check WASD keys
    if keyboard.is_pressed('w'): x_velocity = -1
    elif keyboard.is_pressed('s'): x_velocity = 1
    else: x_velocity = 0
    if keyboard.is_pressed('a'): y_velocity = -1
    elif keyboard.is_pressed('d'): y_velocity = 1
    else: y_velocity = 0
    movement_action = MovementAction(x_velocity, y_velocity)
    # Get mouse position and shoot state (space key)
    mouse_x, mouse_y = pyautogui.position()
    direction = coords2dir(mouse_x, mouse_y)
    is_clicking = keyboard.is_pressed('space')
    shoot_action = ShootAction(direction, is_clicking)
    # Get upgrades
    upgrades_action = UpgradesAction([])
    return ActionSuite(upgrades_action, movement_action, shoot_action)


def countdown_with_message(t, msg):
    for i in range(t):
        print(f"{msg} in {t - i} seconds...", end='\r')
        time.sleep(1)
    return

if __name__ == '__main__':
    print("All dependencies loaded.")
    use_gemini = False
    countdown_with_message(3, "Taking over controls")
    print("\nGame started! Press 'q' to pause AI control for 5 seconds.")

    stream = DataStream(filename='data/stream0/')
    while True:
        t0 = time.time()
        img = get_game_scene()
        actions = get_human_action()
        stream.put(actions, img)
        dt = time.time() - t0
        if dt < 0.1:
            time.sleep(0.1 - dt)
        print(time.time() - t0)
        if keyboard.is_pressed('q'):
            stream.save()
            print("Actions saved to action_stream.json.")
            raise KeyboardInterrupt


    while True:
        dt = 0
        t0 = time.time()
        while is_killed() > 10:
            t = time.time()
            if dt > 0 and t - t0 > dt:
                t0 = t
                score = get_ingame_score()["score"]
                print(f"Score: {score}")
            img = get_game_scene()
            actions = get_action_ai(img)
            apply_action(actions)
            if keyboard.is_pressed('q'):
                release_all_keys()
                print("AI control paused.")
                countdown_with_message(5, "AI will resume controls")
                print("\nGame resumed!")
        release_all_keys()
        if use_gemini:
            stats = get_final_score()
            print(f"Game over! Score: {stats['score']}, Level: {stats['level']}, Time: {stats['time']}")
        else:
            time.sleep(1)
        respawn()
