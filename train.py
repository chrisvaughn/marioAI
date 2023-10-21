import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env = gym_super_mario_bros.make(
    "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
# Loop through each frame in the game
for step in range(100000):
    # Start the game to begin with
    if done:
        # Start the gamee
        env.reset()
    # Do random actions
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    # Show the game on the screen
    env.render()
# Close the game
env.close()
