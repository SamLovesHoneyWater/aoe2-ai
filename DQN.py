import os, random
import torch
import torch.nn as nn

from actions import *
from utils import list_folders

MODEL_SAVE_PATH = 'saved_models/'
#BACKBONE_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'backbone')
#QNET_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'QNet')
#POLICYNET_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'PolicyNet')

class SharedBackbone(nn.Module):
    def __init__(self, backbone_output_size, input_channels=3, input_height=189, input_width=384):
        super(SharedBackbone, self).__init__()
        
        # Convolutional layers to process the screenshot
        '''
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(), #16, 46, 95
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 128, kernel_size=8, stride=4),
            nn.ReLU(), #64, 10, 22
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1024, kernel_size=5, stride=2),
            nn.ReLU(), #64, 3, 9
            nn.BatchNorm2d(1024)
        )
        '''
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2),
            nn.ReLU(), #, 93, 191
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(), #, 45, 94
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(), #, 21, 46
            nn.Conv2d(64, 512, kernel_size=8, stride=4),
            nn.ReLU(), #, 4, 10
            nn.Conv2d(512, 1024, kernel_size=4, stride=2),
            nn.ReLU() #, 1, 4
        )
        #'''
       
        # Calculate the size of flattened features
        self.feature_size = self._get_conv_output((input_channels, input_height, input_width))
        self.fc = nn.Linear(self.feature_size, backbone_output_size)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.reshape(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        # Process the screenshot through conv layers
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        
        return x
    
class QNetHead(nn.Module):
    def __init__(self, backbone_output_size, action_size):  
        super(QNetHead, self).__init__()
        self.fc1 = nn.Linear(backbone_output_size + action_size, 256)
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, backbone_output, actions):
        x = torch.cat([backbone_output, actions], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        reward = self.fc3(x)
        return reward

class PolicyNetHead(nn.Module):
    def __init__(self, backbone_output_size):
        super(PolicyNetHead, self).__init__()
        self.fc = nn.Linear(backbone_output_size, 256)
        
        self.continuous_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.discrete_head1 = nn.Linear(256, 3)  # For -1, 0, 1
        self.discrete_head2 = nn.Linear(256, 3)  # For -1, 0, 1
    
    def forward(self, backbone_output):
        x = torch.relu(self.fc(backbone_output))
        continuous_output = self.continuous_head(x)
        discrete1_output = self.discrete_head1(x)
        discrete2_output = self.discrete_head2(x)
        
        return {
            'continuous': continuous_output,
            'discrete1': discrete1_output,
            'discrete2': discrete2_output
        }


class DQN(nn.Module):
    def __init__(self, backbone_output_size=512):
        super(DQN, self).__init__()
        self.q_backbone = SharedBackbone(backbone_output_size)
        self.q_head = QNetHead(backbone_output_size, 7)
        self.policy_backbone = SharedBackbone(backbone_output_size)
        self.policy_head = PolicyNetHead(backbone_output_size)
    
    def forward_q(self, screen, actions):
        backbone_output = self.q_backbone(screen)
        return self.q_head(backbone_output, actions)
    
    def forward_policy(self, screen):
        backbone_output = self.policy_backbone(screen)
        return self.policy_head(backbone_output)
    
    def get_loss_q(self, screen, actions, target):
        pred = self.forward_q(screen, actions)
        loss = nn.MSELoss()(pred, target)
        return loss
    
    def train_q(self, screen, actions, reward, optimizer):
        loss = self.get_loss_q(screen, actions, reward)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def get_loss_policy(self, screen, targets, mask=None):
        pred = self.forward_policy(screen)
        discrete1_loss = nn.CrossEntropyLoss()(pred['discrete1'], targets['discrete1'])
        discrete2_loss = nn.CrossEntropyLoss()(pred['discrete2'], targets['discrete2'])

        if mask is None:
            continuous_loss = nn.MSELoss()(pred['continuous'], targets['continuous'])
            return continuous_loss + discrete1_loss + discrete2_loss
        
        continuous_loss = nn.MSELoss(reduction='none')(pred['continuous'], targets['continuous'])
        masked_continuous_loss = continuous_loss * mask
        if mask.sum() == 0:  # Prevent nan error caused by division by zero
            return discrete1_loss + discrete2_loss
        
        normalized_continuous_loss = masked_continuous_loss.sum() / mask.sum()
        return normalized_continuous_loss + discrete1_loss + discrete2_loss

    def train_policy_supervised(self, batch, optimizer):
        inputs, targets, continuous_mask = batch
        loss = self.get_loss_policy(inputs, targets, continuous_mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def get_loss_pipeline(self, screen):
        outputs = self.forward_policy(screen)
        continuous_output = outputs['continuous']
        discrete1_output = outputs['discrete1']
        discrete2_output = outputs['discrete2']
        actions = torch.cat([continuous_output, discrete1_output, discrete2_output], dim=1)
        
        q_values = self.forward_q(screen, actions)
        loss = - q_values.mean()
        return loss
    
    def train_policy_reinforce(self, inputs, optimizer):
        loss = self.get_loss_pipeline(inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    
    def save(self, filename):
        filename = os.path.join(MODEL_SAVE_PATH, filename)
        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        filename = os.path.join(MODEL_SAVE_PATH, filename)
        self.load_state_dict(torch.load(filename))


# Example usage:
if __name__ == "__main__":
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = DQN()
    #model.load('model4.pth')
    model = model.to(device)
    
    # List all data
    directory_path = 'data/'
    folders = list_folders(directory_path)
    print(f"Folders in '{directory_path}': {folders}")

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Load data
    for epoch in range(40):
        accuracy = 0
        random.shuffle(folders)
        for folder in folders:
            #print(f"Loading data from '{directory_path}{folder}'")
            try:
                data = DataStream(filename=directory_path + folder, load=True)
            except FileNotFoundError:
                print(f"Folder '{folder}' is corrupt, data.json file not found.")
                continue
            data.shuffle()
            for i, batch in enumerate(data.iterate_batches(batch_size, device)):
                # Train model
                x, y, mask = batch
                sample_input = torch.randn(batch_size, 3, 189, 384).to(device)
                loss = model.train_policy_supervised((x, y, mask), optimizer)
                
        print(f"Epoch {epoch} Training loss: {loss}")
        if epoch % 10 == 0:
            model.save(f'avoider_v0.{epoch//10}.pth')
            lr /= 2
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Save model
    model.save('dqn_test.pth')

