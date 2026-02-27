# SCIAMA GPU Quick Reference Guide

## GPU Allocation - Key Commands

### Interactive GPU Session
```bash
srun --pty --mem=1g --gres=gpu:1 -J interactive -p gpu.q /bin/bash
```

**After connecting, verify your GPU:**
```bash
nvidia-smi                    # Check GPU status
echo $CUDA_VISIBLE_DEVICES    # See assigned GPU IDs
```

---

## Batch Job Submission

### Minimal GPU sbatch Script
```bash
#!/bin/bash
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu
```

### Production-Ready GPU sbatch Script
```bash
#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Set up environment
module purge
# module load cuda/11.x  # Load CUDA if needed
# module load pytorch    # or tensorflow, etc.

# Run your application
cd $SLURM_SUBMIT_DIR
python your_script.py
```

### Submit the job
```bash
sbatch my_gpu_job.sbatch
```

### Monitor job status
```bash
squeue -u $USER        # Check all your jobs
squeue -j <job_id>     # Check specific job
```

### Cancel a job
```bash
scancel <job_id>
```

---

## GPU Hardware Specifications

| Feature | Specification |
|---------|---------------|
| GPU Nodes | gpu01, gpu02 |
| GPUs per Node | 2x 40GB |
| GPU Memory | 40GB each |
| CPU Cores per Node | 128 |
| Max Concurrent Jobs | 4 jobs on gpu.q |
| Jobs per GPU | 1 (exclusive) |
| Partition Name | gpu.q |

---

## GRES (Generic Resource) Syntax

### Format
```
--gres=gpu[:count]
```

### Examples
```bash
#SBATCH --gres=gpu      # Request 1 GPU (default)
#SBATCH --gres=gpu:1    # Request 1 GPU (explicit)
#SBATCH --gres=gpu:2    # Request 2 GPUs
```

---

## NVIDIA Multi-Instance GPU (MIG) Profiles

GPU01 and GPU02 support MIG partitioning for lighter workloads:

| Profile | Memory | Compute Units | Notes |
|---------|--------|---------------|-------|
| 1g.5gb | 5GB | 1 | Light workloads, max 8 per GPU |
| 1g.10gb | 10GB | 1 | Medium workloads |
| 2g.20gb | 20GB | 2 | Heavier workloads |
| 3g.20g | 20GB | 3 | Higher compute demands |
| 4g.20g | 20GB | 4 | Maximum compute allocation |

---

## CUDA/Environment Variables

### Check GPU Assignment
```bash
echo $CUDA_VISIBLE_DEVICES    # Environment variable set by SLURM
```

### Module Management
```bash
module purge                       # Clear all loaded modules
module load system/intel64         # Load system module
module avail | grep -i cuda        # List available CUDA modules
module avail | grep -i gpu         # List available GPU modules
```

---

## Python GPU Job Example

### Script: train_gpu.py
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### sbatch Script: submit_gpu_job.sbatch
```bash
#!/bin/bash
#SBATCH --job-name=pytorch_test
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=gpu_test.log

# Load environment
module purge
# module load pytorch  # if available

python train_gpu.py
```

---

## TensorFlow GPU Job Example

### sbatch Script: submit_tf_job.sbatch
```bash
#!/bin/bash
#SBATCH --job-name=tensorflow_job
#SBATCH --partition=gpu.q
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=tf_job.log

module purge
# module load tensorflow  # if available

python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
python train_model.py
```

---

## Troubleshooting

### Check GPU Availability
```bash
# List all GPU jobs
squeue | grep gpu

# Check node status
sinfo -p gpu.q

# See how many GPUs are free
sinfo -p gpu.q --Node --format="%N %.6D %.6T %.15C"
```

### Verify GPU Allocation in Running Job
```bash
# Monitor GPU usage during job execution
nvidia-smi -l 1    # Refresh every 1 second
```

### CPU-only Fallback
If GPUs are unavailable, you can modify scripts to:
```bash
#SBATCH --partition=cpu.q  # Or appropriate CPU partition
# Comment out #SBATCH --gres=gpu
```

---

## Important Constraints

1. **One job per GPU**: Only 1 job can run on a single GPU at a time
2. **Maximum 4 concurrent GPU jobs**: Limited by 2 GPUs x 2 jobs/GPU
3. **GPU Partition**: Always use `--partition=gpu.q` for GPU jobs
4. **GRES Required**: Must include `--gres=gpu` or `--gres=gpu:N`

---

## Resources

- SCIAMA Documentation: https://sciama.icg.port.ac.uk/sciama-wp/
- NVIDIA MIG Guide: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- SLURM Documentation: https://slurm.schedmd.com/

