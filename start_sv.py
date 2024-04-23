import subprocess

def start_server():
    # Construct the command
    command = r"python .\Backend\start_server.py"

    # Start the server as a separate process
    process = subprocess.Popen(command, shell=True)
    return process

start_server()
