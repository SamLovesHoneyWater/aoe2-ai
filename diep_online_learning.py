import torch
import time, keyboard
import matplotlib.pyplot as plt
import threading

from utils import countdown_with_message
from actions import apply_action, DataStream
from diep_utils import get_game_scene, get_ai_action, is_killed, release_all_keys, get_final_score, get_ingame_score, get_ingame_score_img
from DQN import DQN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def game_loop():
    global latest_score
    i_stream = 41
    while True:
        force_stop = False
        stream = DataStream(filename=f'data/stream{i_stream}/')
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
            stream.put(actions, img, score)

            # Pause AI control
            if keyboard.is_pressed('q'):
                release_all_keys()
                print("Recording session ended by user.")
                force_stop = True
                break

        # Game over (or force stopped)
        release_all_keys()
        stream.save()

        # Train model
        if force_stop:
            print("Training model...")
        else:
            print("Game over! Training model...")
        batch_size = 16
        lr = 0.0001
        training_params = [
                            {'params': model.policy_backbone.parameters()},
                            {'params': model.policy_head.parameters()}
        ]
        critic_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        actor_optimizer = torch.optim.Adam(training_params, lr=lr)
        for epoch in range(2):
            for i, batch in enumerate(stream.iterate_reward_batches(batch_size, device)):
                x, y, rewards = batch
                shift = 5
                # Train critic model
                if x.shape[0] > shift:
                    critic_loss = model.train_q(x[:-shift], y[:-shift], rewards[shift:], critic_optimizer)
                else:
                    critic_loss = model.train_q(x, y, rewards, critic_optimizer)
                # Train policy model
                actor_loss = model.train_policy_reinforce(x, actor_optimizer)
        print(f"Stream {i_stream} (len={len(stream.stream)}): Critic loss = {critic_loss}, Actor loss = {actor_loss}")
        model.save(f'dqn_online_v0.{i_stream}.pth')

        i_stream += 1
        if force_stop:
            # Wait for user in case of force stop
            countdown_with_message(5, "AI will resume controls")
            print("\nGame resumed!")
        else:
            # Otherwise, the character is died and we respawn
            is_killed(do_respawn=True)

def update_score():
    global latest_score
    while True:
        check_score_interval = 1.6
        t0 = time.time()
        img = get_ingame_score_img()
        while not is_killed():
            t1 = time.time()
            if t1 - t0 > check_score_interval > 0:
                with latency_lock:
                    response_latency['gemini'] = t1 - t0
                    latency = response_latency
                t0 = t1
                score = get_ingame_score(img)["score"]
                with score_lock:
                    latest_score = score
                # Update information
                print(f"Score: {score}  |  Latency: {latency}")#, end='\r')
                img = get_ingame_score_img()
        time.sleep(0.2)

if __name__ == '__main__':

    model = DQN()
    model.load('dqn_online_v0.40.pth')
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
