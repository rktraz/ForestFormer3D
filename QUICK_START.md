# Quick Start - Use Regular Directory

## 1. Activate Conda Environment
```bash
source /home/rkaharly/miniconda/etc/profile.d/conda.sh
conda activate forestformer3d
```

## 2. Go to Regular Directory
```bash
cd /home/rkaharly/ForestFormer3D
```

## 3. Run Inference
```bash
./tools/run_inference_single_file.sh test_files/256_leafon.las
```

## 4. Find Results
```bash
ls -lh work_dirs/output/round_1/256_leafon_round1.ply
```
