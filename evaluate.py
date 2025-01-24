# Visualize
#%%
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
import gymnasium as gym

def show_video_of_model(agent, env):
    state, _ = env.reset()
    done = False
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action[0])

    env.close()
    # Save the frames as a GIF instead of MP4
    imageio.mimsave('video.gif', frames, format='GIF', fps=30)
    print("GIF saved successfully.")

def show_video():
    giflist = glob.glob('*.gif')  # Change to look for GIFs
    if len(giflist) > 0:
        gif = giflist[0]
        video = io.open(gif, 'rb').read()  # Open the GIF file
        encoded = base64.b64encode(video)  # Encode the GIF in base64
        display(HTML(data=f'''<img src="data:image/gif;base64,{encoded.decode('ascii')}" style="height: 400px;" autoplay loop>'''))
    else:
        print("Could not find GIF")

# Run the functions
show_video_of_model(agent, env)
show_video()