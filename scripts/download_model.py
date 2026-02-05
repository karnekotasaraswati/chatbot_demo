import os
import urllib.request
import sys

# URL for Phi-2 GGUF
MODEL_URL = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_S.gguf"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "phi-2.Q4_K_S.gguf")

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory: {MODEL_DIR}")

    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return

    print(f"Downloading Phi-2 model to {MODEL_PATH}...")
    print(f"Source: {MODEL_URL}")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = (downloaded / total_size) * 100
        if block_num % 1000 == 0:  # Don't print every block
            sys.stdout.write(f"\rProgress: {percent:.2f}% ({downloaded / (1024*1024):.2f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, report_progress)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nError downloading model: {str(e)}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

if __name__ == "__main__":
    download_model()
