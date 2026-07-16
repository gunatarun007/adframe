from huggingface_hub import HfApi
api = HfApi()
info = api.model_info('Wan-AI/Wan2.1-VACE-14B-diffusers')
total_size = 0
for f in info.siblings:
    if f.size is not None:
        total_size += f.size
        print(f"{f.rfilename}: {f.size / (1024**2):.2f} MB")
print(f"Total size: {total_size / (1024**3):.2f} GB")
