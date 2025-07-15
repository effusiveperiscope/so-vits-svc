def load_state_dict_mismatch(model, state_dict):
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    mismatched_keys = []

    for key in state_dict:
        if key in model_state_dict:
            if state_dict[key].shape == model_state_dict[key].shape:
                filtered_state_dict[key] = state_dict[key]
            else:
                mismatched_keys.append((key, state_dict[key].shape, model_state_dict[key].shape))
        else:
            mismatched_keys.append((key, state_dict[key].shape, None))  # Key not in model

    if mismatched_keys:
        print("Mismatched or missing keys (skipped):")
        for key, shape_ckpt, shape_model in mismatched_keys:
            print(f"{key}: checkpoint shape = {shape_ckpt}, model shape = {shape_model}")

    model.load_state_dict(filtered_state_dict, strict=False)

def load_submodule_prefix(model, prefix : str, state_dict: dict):
    state_dict = {
        k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    load_state_dict_mismatch(model, state_dict)

from omegaconf import OmegaConf
from vits.models import TextEncoder
import torch
hp = OmegaConf.load('configs/base.yaml')
model_to_load = 'vits_pretrain/sovits5.0_bigvgan_mix_v2.pth'
te = TextEncoder(
    hp.vits.ppg_dim,
    hp.vits.vec_dim,
    hp.vits.inter_channels,
    hp.vits.hidden_channels,
    hp.vits.filter_channels,
    2,
    6,
    3,
    0.1,
)
load_submodule_prefix(te, 'module.enc_p.', 
    torch.load(model_to_load, map_location='cpu')['model_g'])
torch.save(te.state_dict(), 'vits_pretrain/svc5_text_encoder.pth')