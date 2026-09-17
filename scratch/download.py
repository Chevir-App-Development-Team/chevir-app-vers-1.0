import urllib.request
import os

def download_video(url, filename):
    print(f"Downloading {url} to {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print("Success!")
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    # download_video('URL', 'salam.mp4')
    pass
