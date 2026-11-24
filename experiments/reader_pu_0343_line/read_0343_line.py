"""Nieuwlaar's awarded PHerc0343 First Letters line (public; surface data/v4/nieuwlaar/sv.zarr, 21 layers): Reader V3 vs
Reader-PU bhand (ScribblePU), no training, same team inference path, both layer directions.
__main__ guard REQUIRED on Windows: DataLoader workers re-import this file (the first version hung without it)."""
import os
import sys
import time

R = r'C:\Users\nikox\Documents\Vesuvius'
O = os.path.join(R, 'data', 'scribblepu', 'u08', 'n343')
SV = os.path.join(R, 'data', 'v4', 'nieuwlaar', 'sv.zarr')
MODELS = [('v3', os.path.join(R, 'release', 'reader-v3', 'checkpoints', 'reader-v3.pth')),
          ('pu_bhand', r'C:\Users\nikox\Documents\antigravity\scribble-pu_hf_card\reader-pu-bhand.pth')]


def main():
    sys.path.insert(0, r'C:\Users\nikox\Documents\antigravity\scribble-pu\experiments\exam1_from_scratch')
    import eval_exam1 as E1
    import logging
    import torch
    from pathlib import Path
    os.makedirs(O, exist_ok=True)
    INF = E1.INF
    for tag, ck in MODELS:
        args = INF.normalize_inference_paths(INF.parse_args([SV, ck, os.path.join(O, '_probe.tif'), '--overlap', '0.5',
                                                             '--blend-mode', 'hann', '--batch-size', '8', '--no-compile',
                                                             '--direction', 'forward', '--num-workers', '2']))
        cm = INF.configure_model(args)
        cm, dev, _ = INF.prepare_model_for_inference(args=args, configured_model=cm)
        logging.getLogger().setLevel(logging.WARNING)
        for d in ('forward', 'reverse'):
            out = os.path.join(O, f'{tag}_{d}.tif')
            if os.path.exists(out):
                continue
            t0 = time.time()
            INF.infer_single_zarr(args=args, configured_model=cm, device=dev, input_zarr=SV, output_tiff=Path(out),
                                  layer_direction=d)
            print(f'[N343 READ] {tag} {d} {time.time() - t0:.0f}s', flush=True)
        del cm
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
