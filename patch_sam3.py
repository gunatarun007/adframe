import os

filepath = './sam3/sam3/model/sam3_multiplex_base.py'
if os.path.exists(filepath):
    print("Patching sam3_multiplex_base.py...")
    content = open(filepath).read()
    target = 'pos_pred_mask_idx = pos_pred_mask.argsort(descending=True)'
    replacement = 'pos_pred_mask_idx = pos_pred_mask.to(torch.int32).argsort(descending=True)'
    if target in content:
        content = content.replace(target, replacement)
        open(filepath, 'w').write(content)
        print("Successfully patched!")
    else:
        print("Target string not found (maybe already patched).")
else:
    print(f"File not found: {filepath}")
