
import platform
import os
import shutil
import subprocess


def setup_xquartz():
    if platform.system() == "Darwin":  # macOS
        print("Setting up XQuartz...")
        try:
            subprocess.run(["curl", "-O", "https://dl.bintray.com/xquartz/downloads/XQuartz-2.7.11.dmg"], check=True)
            subprocess.run(["hdiutil", "attach", "XQuartz-2.7.11.dmg"], check=True)
            subprocess.run(["sudo", "installer", "-pkg", "/Volumes/XQuartz-2.7.11/XQuartz.pkg", "-target", "/"], check=True)
            subprocess.run(["hdiutil", "detach", "/Volumes/XQuartz-2.7.11"], check=True) 
            print("XQuartz installed successfully.")

        except Exception as e:
            print(f"Error installing XQuartz: {e}")
            
    elif platform.system() == "Linux":
        print("XQuartz setup is not required on Linux.")
        # For Linux, you might want to install X11 or similar
        try:
            subprocess.run(["sudo", "apt-get", "install", "-y", "xorg"], check=True)
            print("X11 installed successfully.")
        except Exception as e:
            print(f"Error installing X11: {e}")

    elif platform.system() == "Windows":
        print("Setting up VcXsrv...")
        try:
            subprocess.run(["choco", "install", "vcxsrv"], check=True)
            print("VcXsrv installed successfully.")
        except Exception as e:
            print("To start the Docker image with VcXsrv on Windows, run the following command:")

    else:
        print("XQuartz setup is only required on macOS.")


def configure():
    if platform.system() == "Darwin":  # macOS
        print("Configuring xhost...")
        try:
            subprocess.run(["xhost", "+127.0.0.1"], check=True)
            print("xhost configured successfully.")

        except Exception as e:
            print(f"Error configuring xhost: {e}")

    elif platform.system() == "Windows":  
        # setup VcXsrv for windows  
        print("Configuring VcXsrv for Windows...")
        try:
            subprocess.run(["vcxsrv", "-ac", "-multiwindow", "-clipboard"], check=True)
            print("VcXsrv configured successfully.")
        except Exception as e:
            print(f"Error configuring VcXsrv: {e}")

    else:
        print("xhost configuration is only required on macOS.")

def install_docker_image():
    print("Installing Docker image...")
    try:
        subprocess.run(["sudo", "docker", "build", "-t", "deepplate-img", "."], check=True)
        print("Docker image installed successfully.")
    except Exception as e:
        print(f"Error installing Docker image: {e}")

def check_docker():
    print("Starting setup...")
    # Check if Docker is installed
    if not shutil.which("docker"):
        print("Docker is not installed. Please install Docker first.")
        exit(1)
    # Try to get the Docker daemon running
    try:
        subprocess.run(["docker", "info"], check=True)
        print("Docker daemon is running.")
    except subprocess.CalledProcessError:
        print("Docker daemon is not running. Please start Docker.")
        exit(1)

if __name__ == "__main__":
    check_docker()
    setup_xquartz()
    configure()
    install_docker_image()
    print("Setup completed successfully.")
