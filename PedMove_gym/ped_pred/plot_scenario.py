import imageio
import os

def Create_gif(dir, gif_dir, num_frames, start_frame, scenario_num,
                     early_stop_frame, fps=5):
     
    frames = []
    if early_stop_frame is not None:
        num_frames = early_stop_frame
    for t in range(start_frame, num_frames+1):
        image_dir = os.path.join(dir, f'img_{t}.png')
        image = imageio.imread(image_dir)
        frames.append(image)
    frames.append(frames[-1])
    save_dir = os.path.join(gif_dir, f'scenario{scenario_num}.gif')
    duration = 1000 * 1/fps
    imageio.mimsave(save_dir, # output gif
                frames,          # array of input frames
                duration=duration,
                loop = True)
    

def Create_video(dir, video_dir, num_frames, start_frame, scenario_num,
                 early_stop_frame, fps=25, codec='libx264'):
     
    frames = []
    if early_stop_frame is not None:
        num_frames = early_stop_frame
    for t in range(start_frame, num_frames + 1):
        image_dir = os.path.join(dir, f'img_{t}.png')
        image = imageio.imread(image_dir)
        frames.append(image)
    
    # Duplicate the last frame for smoother video looping
    frames.append(frames[-1])

    save_dir = os.path.join(video_dir, f'scenario{scenario_num}.mp4')
    duration = 1 / fps

    # Write frames to video file
    writer = imageio.get_writer(save_dir, fps=fps, codec=codec)
    for frame in frames:
        writer.append_data(frame)
    writer.close()