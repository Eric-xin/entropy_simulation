import subprocess

def mp4_to_gif(mp4_path, gif_path):
    # Use ffmpeg to convert mp4 to gif
    subprocess.run([
        'ffmpeg', 
        '-i', mp4_path, 
        '-vf', 'fps=10,scale=320:-1:flags=lanczos', 
        '-c:v', 'gif', 
        gif_path
    ])

mp4_to_gif('assets/particle_simulation.mp4', 'assets/particle_simulation.gif')
