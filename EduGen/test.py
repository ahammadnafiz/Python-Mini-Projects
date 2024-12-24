import subprocess
print(subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout)