import subprocess

def assemble_video(animation_path, audio_path, output_path):
    command = f"ffmpeg -i {animation_path} -i {audio_path} -c:v copy -c:a aac -strict experimental {output_path}"
    subprocess.run(command, shell=True)
