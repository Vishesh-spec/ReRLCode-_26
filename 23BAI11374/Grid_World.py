import numpy as np
import random

# 🔹 USER INPUT
grid_size = int(input("Enter grid size (e.g. 5): "))

start_x = int(input("Enter start x: "))
start_y = int(input("Enter start y: "))
start = (start_x, start_y)

goal_x = int(input("Enter goal x: "))
goal_y = int(input("Enter goal y: "))
goal = (goal_x, goal_y)

trap_x = int(input("Enter trap x: "))
trap_y = int(input("Enter trap y: "))
trap = (trap_x, trap_y)

episodes = int(input("Enter number of episodes (e.g. 500): "))

# 🔹 Q-table
q_table = np.zeros((grid_size, grid_size, 4))

# Actions
actions = ["UP", "DOWN", "LEFT", "RIGHT"]

# Hyperparameters
alpha = 0.8
gamma = 0.9
epsilon = 0.2

# 🔹 Move function
def move(state, action):
    x, y = state
    
    if action == 0 and x > 0: x -= 1
    elif action == 1 and x < grid_size - 1: x += 1
    elif action == 2 and y > 0: y -= 1
    elif action == 3 and y < grid_size - 1: y += 1
    
    return (x, y)

# 🔹 Reward function
def get_reward(state):
    if state == goal:
        return 10
    elif state == trap:
        return -10
    else:
        return -1

# 🔹 TRAINING
for episode in range(episodes):
    state = start
    
    while state != goal:
        x, y = state
        
        # Explore vs Exploit
        if random.uniform(0,1) < epsilon:
            action = random.randint(0,3)
        else:
            action = np.argmax(q_table[x, y])
        
        new_state = move(state, action)
        reward = get_reward(new_state)
        
        nx, ny = new_state
        
        # Q-learning update
        q_table[x, y, action] += alpha * (
            reward + gamma * np.max(q_table[nx, ny]) - q_table[x, y, action]
        )
        
        state = new_state

print("\n✅ Training Completed!")

# 🔹 TESTING (Best Path)
state = start
path = [state]

while state != goal:
    x, y = state
    action = np.argmax(q_table[x, y])
    state = move(state, action)
    path.append(state)

# 🔹 OUTPUT
print("\n📍 Optimal Path:")
for step in path:
    print(step)

print("\n🎯 Total Steps:", len(path)-1)