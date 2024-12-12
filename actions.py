import math, json, os
import keyboard, pyautogui
import torch

ALL_KEYS = ['w', 'a', 's', 'd', 'space']

SHOOT_R = 256
SHOOT_X, SHOOT_Y = 960, 550

class UpgradesAction:
    def __init__(self, upgrades: list):
        self.upgrades = upgrades

class MovementAction:
    def __init__(self, x_velocity: int, y_velocity: int):
        self.x_velocity, self.y_velocity = x_velocity, y_velocity

def dir2coords(direction: float):
    return SHOOT_R * math.cos(2 * math.pi * direction) + SHOOT_X, \
        SHOOT_R * math.sin(2 * math.pi * direction) + SHOOT_Y

def coords2dir(x: float, y: float):
    dir = math.atan2(y - SHOOT_Y, x - SHOOT_X) / (2 * math.pi)
    if dir < 0:
        dir += 1
    return dir

class ShootAction:
    def __init__(self, direction: float, shoot: bool):
        self.shoot = shoot
        if shoot:
            self.direction = direction
        else:
            self.direction = None

class ActionSuite:
    def __init__(self, upgrades_action, movement_action, shoot_action):
        self.upgrades_action, self.movement_action, self.shoot_action = upgrades_action, movement_action, shoot_action
    
    def to_dict(self):
        return {
            'upgrades': self.upgrades_action.upgrades,
            'x_velocity': self.movement_action.x_velocity,
            'y_velocity': self.movement_action.y_velocity,
            'do_shoot': self.shoot_action.shoot,
            'direction': self.shoot_action.direction
        }

class DataStream:
    def __init__(self, filename=None, load=False):
        self.stream = []
        if load:
            if filename is None:
                raise ValueError("Filename must be provided to load data.")
            self.load(filename)
        else:
            if filename is None:
                filename = 'data/stream/'
            elif not filename.endswith('/'):
                filename += '/'
            os.makedirs(filename, exist_ok=False)
        self.filename = filename
        
    def put(self, actions: ActionSuite, img):
        img_filename = f"img_{len(self.stream)}.png"
        img.save(self.filename + img_filename)
        self.stream.append((actions, img_filename))
    
    def get_recent(self, n: int, strict=False):
        if len(self.stream) < n:
            if not strict:
                return self.stream
            else:
                raise ValueError(f"Tried to get {n} recent actions, but only {len(self.stream)} available. Set strict=False to get what's available.")
        return self.stream[-n:]

    def save(self):
        data = [
            {
                'actions': actions.to_dict(),
                'img': img_filename
            }
            for actions, img_filename in self.stream
        ]
        with open(self.filename + 'data.json', 'w') as f:
            json.dump(data, f)

    def load(self, filename: str):
        with open(filename + 'data.json', 'r') as f:
            data = json.load(f)
        self.stream = [
            {
                'actions': ActionSuite(
                                    UpgradesAction(actions['upgrades']),
                                    MovementAction(actions['x_velocity'], actions['y_velocity']),
                                    ShootAction(actions['direction'], actions['do_shoot'])
                                ),
                'img': img_filename
            }
            for actions, img_filename in self.stream
        ]


def output_to_actions(outputs):
    # Parse movement action
    x_velocity = torch.argmax(outputs['discrete1'], dim=1)
    y_velocity = torch.argmax(outputs['discrete2'], dim=1)
    x_velocity = x_velocity.cpu().detach().numpy()[0]
    y_velocity = y_velocity.cpu().detach().numpy()[0]
    movement_action = MovementAction(x_velocity, y_velocity)
    # Parse shoot action
    shoot_dir = outputs['continuous']
    shoot_dir = shoot_dir.cpu().detach().numpy()[0][0]  # Extract float value from cuda tensor
    do_shoot = torch.argmax(outputs['onehot'], dim=1)  # Convert onehot to bool
    do_shoot = do_shoot.cpu().detach().numpy()[0] == 1
    shoot_action = ShootAction(shoot_dir, do_shoot)
    # Upgrade dummy
    upgrades_action = UpgradesAction([])
    return ActionSuite(upgrades_action, movement_action, shoot_action)

def apply_action(actions: ActionSuite):
    # Apply upgrades
    for key in actions.upgrades_action.upgrades:
        keyboard.press(key)
        keyboard.release(key)
    # Apply movements
    if actions.movement_action.x_velocity == 0:
        keyboard.release('a')
        keyboard.release('d')
    elif actions.movement_action.x_velocity == 1:
        keyboard.press('d')
        keyboard.release('a')
    else:
        keyboard.press('a')
        keyboard.release('d')
    if actions.movement_action.y_velocity == 0:
        keyboard.release('w')
        keyboard.release('s')
    elif actions.movement_action.y_velocity == 1:
        keyboard.press('s')
        keyboard.release('w')
    else:
        keyboard.press('w')
        keyboard.release('s')
    # Apply shooting
    if actions.shoot_action.shoot:
        x, y = dir2coords(actions.shoot_action.direction)
        pyautogui.moveTo(x, y)
        keyboard.press('space')
    else:
        keyboard.release('space')
