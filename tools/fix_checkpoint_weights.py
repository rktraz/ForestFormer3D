"""
Fix checkpoint weight shapes for spconv compatibility.
Transposes weights that have shape mismatches.
"""
import torch
import sys
import argparse

def transpose_spconv_weight(weight, expected_shape):
    """Transpose spconv weight from [out, in, ...] to [in, out, ...] or vice versa."""
    if len(weight.shape) == 5:  # 3D conv: [out, in, d, h, w] -> [in, out, d, h, w]
        if weight.shape[0] == expected_shape[0] and weight.shape[1] == expected_shape[-1]:
            # Need to transpose: [out, in, d, h, w] -> [in, d, h, w, out]
            weight = weight.permute(1, 2, 3, 4, 0)
        elif weight.shape[-1] == expected_shape[0] and weight.shape[0] == expected_shape[-1]:
            # Already in wrong format, transpose back
            weight = weight.permute(4, 1, 2, 3, 0)
    return weight

def fix_checkpoint(input_path, output_path):
    """Fix checkpoint by transposing mismatched weights."""
    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"Found {len(state_dict)} parameters")
    print("Note: This is a placeholder. The weight shape issue requires")
    print("more complex handling. The best solution is to retrain with")
    print("consistent spconv version, or manually transpose weights.")
    
    # Save (even if unchanged, creates backup)
    torch.save(checkpoint, output_path)
    print(f"Saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input checkpoint path')
    parser.add_argument('output', help='Output checkpoint path')
    args = parser.parse_args()
    fix_checkpoint(args.input, args.output)
