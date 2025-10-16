# DragonMAP HPC Environment Setup

This document outlines the setup of a persistent GPU-enabled Singularity and Conda environment for the DragonMAP project on NYU HPC Cloud Bursting.

## 1. Runtime Access
Request an interactive node on the HPC Cloud Bursting cluster.

```bash
ssh burst
srun --account= <Account_Name_Here> --partition=c12m85-a100-1 --gres=gpu:1 --time=04:00:00 --pty /bin/bash
```

For CPU-only work:

```bash
srun --account= <Account_Name_Here> --partition=interactive --time=04:00:00 --pty /bin/bash
```

## 2. Project and Overlay
Create the project directory and unpack a writable overlay for persistence.

```bash
mkdir -p /scratch/$USER/dragonmap
cd /scratch/$USER/dragonmap
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz
```

## 3. Singularity Base Image
Use the CUDA 12.6 + cuDNN 9.5 Ubuntu 22.04 image.

```bash
SIF=/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
singularity exec --nv --overlay overlay-15GB-500K.ext3:rw $SIF /bin/bash
```

## 4. Miniforge Installation
Install Miniforge inside the container.

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
```

## 5. Environment Creation
Initialize Conda and create the environment.

```bash
source /ext3/miniforge3/etc/profile.d/conda.sh
conda create -y -n dragonmap python=3.10
conda activate dragonmap
conda install -y -c conda-forge numpy scipy pandas scikit-learn tqdm requests pyyaml
```

## 6. PyTorch with CUDA
Install PyTorch with bundled CUDA libraries.

```bash
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.6
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 7. Environment Activation Script
Create `/ext3/env.sh` to automatically activate the environment when the container is opened.

```bash
#!/bin/bash
unset -f which 2>/dev/null || true
eval "$(/ext3/miniforge3/bin/conda shell.bash hook)"
export PATH=/ext3/miniforge3/bin:$PATH
conda activate dragonmap 2>/dev/null || source /ext3/miniforge3/bin/activate dragonmap
export PATH=/ext3/miniforge3/envs/dragonmap/bin:$PATH
hash -r
for d in $HOME/.local/lib/python*/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH}"
done
for d in /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH}"
done
```

## 8. Startup Script
Create a reusable script to launch the environment with one command.

```bash
#!/bin/bash
cd /scratch/$USER/dragonmap
SIF=/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
singularity exec --nv --overlay overlay-15GB-500K.ext3:rw "$SIF" /bin/bash -lc "source /ext3/env.sh; bash"
```

Make it executable:

```bash
chmod +x /scratch/$USER/dragonmap/start_dragonmap.sh
```

## 9. Alias for Quick Launch
Add an alias for easier access.

```bash
echo "alias dragonmap='bash /scratch/$USER/dragonmap/start_dragonmap.sh'" >> ~/.bashrc
source ~/.bashrc
```

Run the environment with:
```bash
dragonmap
```

## 10. Jupyter Kernel Integration
To use the environment as a Jupyter kernel, create a wrapper script and kernel spec.

### Wrapper Script
`/scratch/$USER/dragonmap/dragonmap_kernel.sh`

```bash
#!/bin/bash
SIF=/scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/$USER/dragonmap/overlay-15GB-500K.ext3
exec singularity exec --nv --overlay "${OVERLAY}:rw" "$SIF" /bin/bash -lc   "source /ext3/env.sh; python -m ipykernel_launcher -f "$1""
```

### Kernel Specification
`~/.local/share/jupyter/kernels/dragonmap-singularity/kernel.json`

```json
{
  "argv": ["bash", "/scratch/$USER/dragonmap/dragonmap_kernel.sh", "{connection_file}"],
  "display_name": "Python (dragonmap via Singularity)",
  "language": "python"
}
```

Refresh Jupyter and select Python (dragonmap via Singularity).

## 11. Verification
Run the following inside Jupyter or terminal.

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

Expected output:

```
2.9.0+cu126
CUDA: True
NVIDIA A100-SXM4-40GB
```

## 12. Further Resources and Troubleshooting


### Official NYU HPC Documentation
- [HPC Access Guide](https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc): Steps for logging in, managing credentials, and accessing Greene and Cloud Bursting.  
- [Cloud Bursting OOD Portal](https://ood-burst-001.hpc.nyu.edu): Web interface for launching Jupyter, terminal sessions, and monitoring runtime.  
- [Singularity with Miniconda](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda): Instructions for overlays and Conda within Singularity.  
- Overlay templates: `/share/apps/overlay-fs-ext3`  
- Base images: `/share/apps/images`

---

### GPU Runtime and CUDA References
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch CUDA Compatibility Matrix](https://pytorch.org/get-started/previous-versions/)
- [Google Cloud Spot Instances](https://cloud.google.com/compute/docs/instances/spot)

---

### Common Issues and Fixes

| **Symptom** | **Likely Cause** | **Recommended Fix** |
|--------------|------------------|----------------------|
| `python: command not found` inside Singularity | Conda not initialized properly | Re-source `/ext3/env.sh` or ensure `conda shell.bash hook` is used |
| `ImportError: libcusparseLt.so.0` | PyTorch pip wheel missing CUDA runtime | Install via conda: `conda install -c pytorch -c nvidia pytorch-cuda=12.6` |
| Jupyter kernel missing | Kernel not registered inside Singularity | Run `python -m ipykernel install --user --name dragonmap` within container |
| `[Errno 2] No such file or directory: '/ext3/.../python'` | Kernel registered outside container | Use wrapper kernel with `dragonmap_kernel.sh` |
| GPU not detected | Missing `--nv` flag when starting Singularity | Always include `--nv` in all Singularity commands |
| Package installation slow or failing | Overlay storage near full | Use 15 GB overlay and clear caches: `conda clean -a` |

---