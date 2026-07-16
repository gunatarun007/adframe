from huggingface_hub import HfApi
api = HfApi()
total_size = 0
for f in api.list_repo_tree('Wan-AI/Wan2.1-VACE-14B-diffusers', recursive=True):
    if hasattr(f, 'size') and f.size is not None:
        total_size += f.size
        print(f"{f.rfilename}: {f.size / (1024**2):.2f} MB")
print(f"Total size: {total_size / (1024**3):.2f} GB")
