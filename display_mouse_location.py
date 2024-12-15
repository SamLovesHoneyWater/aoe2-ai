import pyautogui
import keyboard
import time

try:
    while True:
        x, y = pyautogui.position()
        print(f"Mouse position: ({x}, {y})")
        print(keyboard.is_pressed('h'))
        time.sleep(1)
except KeyboardInterrupt:
    print("Program terminated.")
# -1920, 0
# -385, 959