import os, random
import torch
import torch.nn as nn

from actions import *
from utils import list_folders

MODEL_SAVE_PATH = 'saved_models/'

class GameAI(nn.Module):
    def __init__(self, input_channels=3, input_height=189, input_width=384):
        super(GameAI, self).__init__()
        
        # Convolutional layers to process the screenshot
        #'''
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(), #16, 46, 95
            nn.Conv2d(32, 128, kernel_size=8, stride=4),
            nn.ReLU(), #64, 10, 22
            nn.Conv2d(128, 1024, kernel_size=5, stride=2),
            nn.ReLU() #64, 3, 9
        )
        '''
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=4, stride=2),
            nn.ReLU(), #8, 93, 191
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(), #16, 45, 94
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(), #32, 21, 46
            nn.Conv2d(32, 256, kernel_size=8, stride=4),
            nn.ReLU(), #256, 4, 10
            nn.Conv2d(256, 512, kernel_size=4, stride=2),
            nn.ReLU() #512, 1, 4
        )
        '''


        
        # Calculate the size of flattened features
        self.feature_size = self._get_conv_output((input_channels, input_height, input_width))
        
        # Fully connected layers for different outputs
        self.fc_shared = nn.Linear(self.feature_size, 512)
        
        # Specific output layers
        self.onehot_head = nn.Sequential(
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        
        self.continuous_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        self.discrete_head1 = nn.Linear(512, 3)  # For -1, 0, 1
        self.discrete_head2 = nn.Linear(512, 3)  # For -1, 0, 1

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
        
        # Shared features
        shared_features = torch.relu(self.fc_shared(x))
        
        # Generate different outputs
        onehot_output = self.onehot_head(shared_features)
        continuous_output = self.continuous_head(shared_features)
        discrete1_output = self.discrete_head1(shared_features)
        discrete2_output = self.discrete_head2(shared_features)
        
        return {
            'onehot': onehot_output,
            'continuous': continuous_output,
            'discrete1': discrete1_output,
            'discrete2': discrete2_output
        }
    
    def training_step(self, batch, optimizer):
        inputs, targets, continuous_mask = batch
        outputs = self(inputs)
        
        # Calculate losses for each output
        onehot_loss = nn.CrossEntropyLoss()(outputs['onehot'], targets['onehot'])
        mse_loss = nn.MSELoss(reduction='none')(outputs['continuous'], targets['continuous'])
        discrete1_loss = nn.CrossEntropyLoss()(outputs['discrete1'], targets['discrete1'])
        discrete2_loss = nn.CrossEntropyLoss()(outputs['discrete2'], targets['discrete2'])

        masked_mse_loss = mse_loss * continuous_mask
        if mask.sum() == 0:
            total_loss = onehot_loss + discrete1_loss + discrete2_loss
        else:
            normalized_mse_loss = masked_mse_loss.sum() / mask.sum()
            total_loss = onehot_loss + normalized_mse_loss + discrete1_loss + discrete2_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
    
    def save(self, filename):
        filename = os.path.join(MODEL_SAVE_PATH, filename)
        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        filename = os.path.join(MODEL_SAVE_PATH, filename)
        self.load_state_dict(torch.load(filename))

def toy():
    # Example input
    batch_size = 16
    sample_input = torch.randn(batch_size, 3, 189, 384).to(device)
    
    # Get outputs
    outputs = model(sample_input)
    print(f"Using device: {device}")
    print("Output shapes:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
        print(value)
    
    # Create sample labels
    sample_labels = {
        'onehot': torch.tensor([[1., 0.] for i in range(batch_size)]).to(device),  # Example binary classification
        'continuous': torch.tensor([[0.5] for i in range(batch_size)]).to(device),  # Example value between 0 and 1
        'discrete1': torch.tensor([1 for i in range(batch_size)]).to(device),  # Example class index (0, 1, or 2)
        'discrete2': torch.tensor([2 for i in range(batch_size)]).to(device)   # Example class index (0, 1, or 2)
    }
    #/fix  Discrete labels should be probabilities instead of 
    # Create optimizer
    lr = 0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    last_loss = 0
    for i in range(1000):
        # Train for one step
        loss = model.training_step((sample_input, sample_labels), optimizer)
        if loss == last_loss:
            lr *= 0.9
            print(lr)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        last_loss = loss
        print(f"Training loss: {loss}")



# Example usage:
if __name__ == "__main__":
    batch_size = 16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = GameAI()
    #model.load('model4.pth')
    model = model.to(device)
    #toy()
    
    # List all data
    directory_path = 'data/'
    folders = list_folders(directory_path)
    print(f"Folders in '{directory_path}': {folders}")

    lr = 0.00001
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
                loss = model.training_step((x, y, mask), optimizer)
                if False and i % (200 // batch_size) == 0:
                    print(loss)
                
        print(f"Epoch {epoch} Training loss: {loss}")
        if epoch % 10 == 0:
            model.save(f'avoider_v0.{epoch//10}.pth')
            lr /= 2
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Save model
    model.save('avoider_v0_final.pth')

