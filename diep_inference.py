import torch
import time, keyboard
import matplotlib.pyplot as plt

from utils import countdown_with_message
from actions import apply_action
from diep_utils import get_game_scene, get_ai_action, is_killed, release_all_keys, get_final_score, get_ingame_score
from DQN import DQN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    model = DQN()
    model.load('dqn_test.pth')
    model = model.to(device)

    print("All dependencies loaded.")
    use_gemini = False

    countdown_with_message(10, "Taking over controls")
    print("\nGame started! Press 'q' to pause AI control for 5 seconds.")
    while True:
        dt = 0
        t0 = time.time()
        while not is_killed():
            t = time.time()
            if dt > 0 and t - t0 > dt and use_gemini:
                t0 = t
                score = get_ingame_score()["score"]
                print(f"Score: {score}")
            img = get_game_scene()
            actions = get_ai_action(img, model)
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
        is_killed(do_respawn=True)
