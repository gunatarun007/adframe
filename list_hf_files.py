from huggingface_hub import HfApi
api = HfApi()
info = api.model_info('Wan-AI/Wan2.1-VACE-14B-diffusers')
total_size = 0
for f in info.siblings:
    size = f.size
    if hasattr(f, 'lfs') and f.lfs is not None:
        size = f.lfs.size
    if size is not None:
        total_size += size
        print(f"{f.rfilename}: {size / (1024**2):.2f} MB")
print(f"Total size: {total_size / (1024**3):.2f} GB")
