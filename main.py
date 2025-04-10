from src.run_video_file_stream import run_video_stream
from src.run_live_stream import run_live_stream

def main():
    options = {
        '1': ("Running live stream detection...", run_live_stream),
        '2': ("Running file stream detection...", run_video_stream)
    }

    print("Select an option to run:")
    print("[1] Live stream")
    print("[2] File stream")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice in options:
        message, action = options[choice]
        print(message)
        action()
    else:
        print("Invalid input. Please enter '1' for live stream or '2' for file stream.")

if __name__ == "__main__":
    main()