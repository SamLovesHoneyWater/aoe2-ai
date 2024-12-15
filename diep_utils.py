from PIL import Image
import numpy as np
import mss, time, keyboard, pyautogui
import torch

from gemini_utils import gemini_parse_image
from actions import ALL_KEYS, UpgradesAction, MovementAction, ShootAction, coords2dir, ActionSuite, output_to_actions

GAMESCREEN_X0, GAMESCREEN_Y0 = 0, 75
GAMESCREEN_X1, GAMESCREEN_Y1 = 1920, 1024

image_paths = ['sample_images/continue_button_1.png', 'sample_images/continue_button_2.png']
label_imgs = [np.array(Image.open(image_path)) for image_path in image_paths]

def release_all_keys():
    for key in ALL_KEYS:
        keyboard.release(key)
    return

def get_screen(x0, y0, x1, y1):
    with mss.mss() as sct:
        monitor = {"top": y0, "left": x0, "width": x1 - x0, "height": y1 - y0}
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    return img

def get_game_scene():
    img = get_screen(GAMESCREEN_X0, GAMESCREEN_Y0, GAMESCREEN_X1, GAMESCREEN_Y1)
    #img = img.convert('L')
    k = .2
    img = img.resize((int((GAMESCREEN_X1 - GAMESCREEN_X0) * k), int((GAMESCREEN_Y1 - GAMESCREEN_Y0) * k)), Image.Resampling.LANCZOS)
    return img

def is_killed(do_respawn=False):
    x0, y0 = 940, 725
    x1, y1 = 1103, 775
    img = get_screen(x0, y0, x1, y1)
    img_np = np.array(img)
    diff = np.abs(label_imgs[0] - img_np)
    avg_diff_1 = np.mean(diff)
    if avg_diff_1 < 10:
        if do_respawn:
            respawn(x0, y0, x1, y1)
        return True
    x0, y0 = 940, 770
    x1, y1 = 1103, 820
    img = get_screen(x0, y0, x1, y1)
    img_np = np.array(img)
    diff = np.abs(label_imgs[1] - img_np)
    avg_diff_2 = np.mean(diff)
    if avg_diff_2 < 10:
        if do_respawn:
            respawn(x0, y0, x1, y1)
        return True
    return False
    
def respawn(x0, y0, x1, y1):
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

def get_action_ai(img, policy_model):
    img = np.array(img)
    img = img.reshape(1, 3, img.shape[0], img.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = torch.from_numpy(img).float().to(device)
    outputs = policy_model(img_tensor)
    actions = output_to_actions(outputs)
    return actions

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
