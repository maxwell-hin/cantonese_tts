import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS

# Check CUDA availability and compatibility
try:
    if torch.cuda.is_available():
        # Test CUDA functionality
        torch.cuda.init()
        device = "cuda"
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("CUDA not available")
except Exception as e:
    print(f"CUDA error: {e}")
    device = "cpu"
    print("Falling back to CPU")

print(f"Using device: {device}")

# Load models on the determined device
model = ChatterboxTTS.from_pretrained(device=device)
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)


chinese_text = "你Sunday嘅Booking已經full咗，如果你仲想訂我地tornado餐廳，可以畀返你一個Alternative Booking Time，10月28號星期一11:30AM"
wav_chinese = multilingual_model.generate(chinese_text, language_id="zh")
ta.save("test-chinese.wav", wav_chinese, model.sr)



# Cantonese TTS Application

if __name__ == "__main__":
    pass