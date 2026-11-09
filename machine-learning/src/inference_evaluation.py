import bigvgan
import sys
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
from torch import load, no_grad, inference_mode
from soundfile import write
script_root = Path(__file__).resolve().parent
sys.path.insert(0, str(script_root))

from models.transformer import TransformerModel
from dataset.transformer_dataset import SSIDataset, pad_collate_fn
from utils.utils_transformer import setup_manual_seed, setup_torch
from utils.hyperparameters_transformer import Hyperparameters

PRETRAINED_MODEL_PATH = "/home/jonges/ssi-machine-learning/experiments/vub-smartphone/unvoiced/model_final.pt"
DATA_PATH = Path("transformer_data")
UNVOICED_PATH = DATA_PATH / "unvoiced_samples"
VOICED_PATH = DATA_PATH / "voiced_samples"
TEXTGRID_PATH = DATA_PATH / "textgrids"
OUTPUT_AUDIO_PATH = Path("generated_audio/smartphone_unvoiced/voiced")
OUTPUT_AUDIO_PATH.mkdir(exist_ok=True)

hyperparameters = Hyperparameters()
hyperparameters.batch_size = 1

setup_manual_seed(hyperparameters.seed)
setup_torch(hyperparameters.device)

hyperparameters.device = 'cuda:0'

# Load the dataset
dataset = SSIDataset(hyperparameters, debug=False)
dataset._load_data(VOICED_PATH, UNVOICED_PATH, TEXTGRID_PATH)
dataloader = DataLoader(dataset, pin_memory=isinstance(hyperparameters.device, str) and hyperparameters.device.startswith('cuda'), batch_size=hyperparameters.batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=hyperparameters.num_workers)

# Load the pre-trained model checkpoint
print(f"Loading model checkpoint from {PRETRAINED_MODEL_PATH}")
model = TransformerModel(input_dim_lp=34, model_dim=512, num_heads=4, num_layers=4, dropout=0.3, loss_weight_init=hyperparameters.phoneme_loss_weight_init).to(hyperparameters.device)
checkpoint = load(PRETRAINED_MODEL_PATH, map_location=hyperparameters.device)
model.load_state_dict(checkpoint['model_state_dict'])

bigvgan_model = bigvgan.BigVGAN.from_pretrained(
    "nvidia/bigvgan_v2_44khz_128band_256x",
    use_cuda_kernel=False,
    proxies=None,
    resume_download=False,
)
bigvgan_model.remove_weight_norm()
bigvgan_model = bigvgan_model.eval().to(hyperparameters.device)

print(bigvgan_model.h)

# Load 5 random batches from the dataloader and evaluate the model
model.eval()
with no_grad():
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        lp_input = batch['voiced']['lp'].to(hyperparameters.device)
        pha_input = batch['voiced']['pha'].to(hyperparameters.device)
        amp_input = batch['voiced']['amp'].to(hyperparameters.device)
        dop_input = batch['voiced']['dop'].to(hyperparameters.device)
        lengths = batch['voiced']['lengths'].to(hyperparameters.device)

        gt = batch['gt']['data'].to(hyperparameters.device)
        gt_lengths = batch['gt']['lengths'].to(hyperparameters.device)
        gt = gt.transpose(1, 2)  # Transpose to [batch_size, features, time] for BigVGAN

        # Forward pass: LP + UltraSR features (pha, amp, dop)
        outputs_seq, output_lengths, phonemes = model(lp_input, pha_input, amp_input, dop_input, lengths=lengths)

        outputs_seq = outputs_seq.transpose(1, 2)  # Transpose to [batch_size, features, time] for BigVGAN
        # Generate audio using BigVGAN
        with inference_mode():
            wav_gen = bigvgan_model(outputs_seq)
        wav_gen_cpu = wav_gen.cpu()

        final_audios = []
        for k in range(wav_gen_cpu.size(0)):
            # Calculate how many audio samples the true frames equal
            true_audio_length = gt_lengths[k].item() * 256

            # Select k-th example: wav_k has shape [1, T] or [T]
            wav_k = wav_gen_cpu[k]
            # If shape is [1, T], squeeze channel dim
            if wav_k.dim() == 2 and wav_k.size(0) == 1:
                wav_k = wav_k.squeeze(0)

            # Slice to remove trailing padded noise
            clean_audio = wav_k[:true_audio_length]

            # Convert to numpy array (float32) in shape (n_samples,) or (n_samples, channels)
            try:
                audio_np = clean_audio.numpy()
            except Exception:
                audio_np = clean_audio.detach().cpu().numpy()

            final_audios.append(audio_np)
        
        # # Save the generated audio for manual listening
        for j, audio_np in enumerate(final_audios):
            output_file = OUTPUT_AUDIO_PATH / f"Unvoiced_voiced_{i+1}_sample_{j+1}.wav"
            # Ensure numpy dtype and pass string path
            audio_np = np.asarray(audio_np, dtype=np.float32)

            # If audio is shape (channels, samples), transpose to (samples, channels)
            if audio_np.ndim == 2 and audio_np.shape[0] <= 8 and audio_np.shape[0] < audio_np.shape[1]:
                audio_np = audio_np.T

            write(str(output_file), audio_np, hyperparameters.sample_rate)
            print(f"Saved generated audio to {output_file}")