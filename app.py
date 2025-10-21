import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS

# Set device to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)


chinese_text = "你Sunday嘅Booking已經full咗，如果你仲想訂我地tornado餐廳，可以畀返你一個Alternative Booking Time，10月28號星期一11:30AM"
wav_chinese = multilingual_model.generate(chinese_text, language_id="zh")
ta.save("test-chinese.wav", wav_chinese, model.sr)



# Cantonese TTS Application

if __name__ == "__main__":
    pass