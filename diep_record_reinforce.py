import torch
import time, keyboard
import matplotlib.pyplot as plt
import threading

from utils import countdown_with_message
from actions import apply_action, DataStream
from diep_utils import get_game_scene, get_ai_action, is_killed, release_all_keys, get_final_score, get_ingame_score
from DQN import DQN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def game_loop():
    global latest_score
    i = 0
    while True:
        stream = DataStream(filename=f'data/stream{i}/')
        play_interval = 0.2
        t0 = time.time()
        while not is_killed():
            # Play at a fixed rate
            t1 = time.time()
            if t1 - t0 > play_interval > 0:
                with latency_lock:
                    response_latency['gameplay'] = t1 - t0
                t0 = t1

            # Update score
            with score_lock:
                score = latest_score

            # Play the game
            img = get_game_scene()
            actions = get_ai_action(img, model)
            apply_action(actions)
            print(actions)
            stream.put(actions, img, score)

            # Pause AI control
            if keyboard.is_pressed('q'):
                release_all_keys()
                print("AI control paused.")
                stream.save()
                countdown_with_message(5, "AI will resume controls")
                print("\nGame resumed!")

        # Game over
        release_all_keys()
        i += 1
        stream.save()
        stats = get_final_score()
        print(f"Game over! Score: {stats['score']}, Level: {stats['level']}, Time: {stats['time']}")
        is_killed(do_respawn=True)

def update_score():
    global latest_score
    while True:
        check_score_interval = 1.6
        t0 = time.time()
        while not is_killed():
            t1 = time.time()
            if t1 - t0 > check_score_interval > 0:
                with latency_lock:
                    response_latency['gemini'] = t1 - t0
                    latency = response_latency
                t0 = t1
                score = get_ingame_score()["score"]
                with score_lock:
                    latest_score = score
                # Update information
                print(f"Score: {score}  |  Latency: {latency}")#, end='\r')

if __name__ == '__main__':

    model = DQN()
    #model.load('model1_v1.pth')
    model = model.to(device)

    print("All dependencies loaded.")

    countdown_with_message(10, "Taking over controls")
    print("\nGame started! Press 'q' to pause AI control for 5 seconds.")
    
    latest_score = None
    response_latency = {'gameplay': -1, 'gemini': -1}
    score_lock = threading.Lock()
    latency_lock = threading.Lock()

    # Start threads
    score_reader_thread = threading.Thread(target=update_score, daemon=True)
    game_loop_thread = threading.Thread(target=game_loop, daemon=True)

    score_reader_thread.start()
    game_loop_thread.start()

    # Keep the main thread alive
    while True:
        time.sleep(1)
