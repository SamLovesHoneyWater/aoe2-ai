from DQN import *
from actions import DataStream

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda'

batch_size = 16

model = DQN()
#model.load('critic/v0.0.0.pth')
model = model.to(device)

directory_path = 'data/'
folders = list_folders(directory_path)
print(f"Folders in '{directory_path}': {folders}")

lr = 0.00001
training_params = [
                    {'params': model.policy_backbone.parameters()},
                    {'params': model.policy_head.parameters()}
]
critic_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
actor_optimizer = torch.optim.Adam(training_params, lr=lr)

# Load data
for epoch in range(10):
    accuracy = 0
    random.shuffle(folders)
    critic_losses = []
    for folder in folders:
        try:
            data = DataStream(filename=directory_path + folder, load=True)
        except FileNotFoundError:
            print(f"Folder '{folder}' is corrupt, data.json file not found.")
            continue
        data.shuffle()
        for i, batch in enumerate(data.iterate_reward_batches(batch_size, device)):
            shift = 5
            x, y, rewards = batch
                # Train critic model
            if x.shape[0] > shift:
                critic_loss = model.train_q(x[:-shift], y[:-shift], rewards[shift:], critic_optimizer)
            else:
                critic_loss = model.train_q(x, y, rewards, critic_optimizer)
            # Train with actor-critic
            #actor_loss = model.train_policy_reinforce(x, actor_optimizer)
            critic_losses.append(critic_loss)
    critic_loss = sum(critic_losses) / len(critic_losses)
    print(f"Epoch {epoch}: Critic loss = {critic_loss}, Actor loss = {None}")
    save_freq = 5
    if epoch % save_freq == 0:
        model.save(f'critic/v0.0.{epoch//save_freq}.pth')
        lr /= 2
        critic_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        actor_optimizer = torch.optim.Adam(training_params, lr=lr)

# Save model
model.save('critic/v0.1.0.pth')