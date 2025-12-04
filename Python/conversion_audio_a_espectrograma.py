import os
import numpy as np
import librosa
from PIL import Image

SAMPLE_RATE = 16000
DURATION = 1
SAMPLES = SAMPLE_RATE * DURATION

input_root = "/content/drive/MyDrive/Certamen_Softcomputing/audios/data"
output_root = "/content/drive/MyDrive/Certamen_Softcomputing/espec16_tts"

for person in os.listdir(input_root):
    person_in = os.path.join(input_root, person)
    person_out = os.path.join(output_root, person)
    os.makedirs(person_out, exist_ok=True)

    for fname in os.listdir(person_in):
        if fname.endswith(".mp3"):
            fpath = os.path.join(person_in, fname)
            audio, sr = librosa.load(fpath, sr=SAMPLE_RATE)


            if len(audio) < SAMPLES:
                audio = np.pad(audio, (0, SAMPLES - len(audio)))
            else:
                audio = audio[:SAMPLES]

            stft = librosa.stft(audio, n_fft=1024, hop_length=256, win_length=1024)
            spect = np.abs(stft)**2

            spect_db = librosa.power_to_db(spect, ref=np.max)
            spect_img = spect_db - spect_db.min()
            spect_img = spect_img / spect_img.max()
            spect_img = (spect_img * 255).astype(np.uint8)
            img = Image.fromarray(spect_img)
            img = img.resize((16, 16), Image.BICUBIC)
            img = img.convert("L")  # grayscale

            out_path = os.path.join(person_out, fname.replace(".mp3", ".png"))
            img.save(out_path, format="PNG", compress_level=0)