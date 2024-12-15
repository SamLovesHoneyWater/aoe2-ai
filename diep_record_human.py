import torch
import time, keyboard
import matplotlib.pyplot as plt

from actions import DataStream
from diep_utils import get_game_scene, get_human_action

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    i = 0
    while True:
        time.sleep(0.5)
        print("Press 'q' to start recording actions.")
        while not keyboard.is_pressed('q'):
            time.sleep(0.05)
        stream = DataStream(filename=f'data/stream{i}/')

        time.sleep(0.5)
        #countdown_with_message(10, "Taking over controls")
        print("\nGame started! Press 'q' to pause recording.")

        while True:
            t0 = time.time()
            img = get_game_scene()
            actions = get_human_action()
            stream.put(actions, img)
            dt = time.time() - t0
            if dt < 0.1:
                time.sleep(0.1 - dt)
            #print(time.time() - t0)
            if keyboard.is_pressed('q'):
                stream.save()
                print("Actions saved to action_stream.json.")
                i += 1
                break
