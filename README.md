# DPVO Deployment Guide

This document summarizes how to install, run, and containerize the DPVO pipeline, with special attention to automotive front-camera footage and reproducible outputs.

---

## 1. Native Installation

Tested on Ubuntu 20.04/22.04 with CUDA 11.x/12.x and Python 3.10.

1. **Clone the repository**
   ```bash
   git clone https://github.com/princeton-vl/DPVO.git --recursive
   cd DPVO
   ```
2. **Create the Conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate dpvo
   ```
3. **Install DPVO and its dependencies**
   ```bash
   wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
   unzip eigen-3.4.0.zip -d thirdparty
   pip install .
   ```
4. **Download pretrained model**
   ```bash
   ./download_model.sh
   ```

### Example: Tesla Front Camera

The repository ships with reference intrinsics for the Tesla front camera in `calib/tesla.txt`. Run DPVO on a front-camera video with:

```bash
python run.py \
  --imagedir=/absolute/path/to/front_camera.mp4 \
  --calib=calib/tesla.txt \
  --stride=5 \
  --plot \
  --save_camera_poses \
  --save_motion \
  --save_trajectory
```

All exports are written to `outputs/<run_name>`. Omit `--name` to auto-generate a timestamped directory.

### Adapting to Other Cameras

Each calibration file in `calib/` contains four whitespace-separated values per line: `fx fy cx cy`. To support a new camera:

1. Estimate intrinsic parameters (e.g., with an OpenCV calibration routine).
2. Add a new text file under `calib/` containing the four intrinsics.
3. Invoke `run.py` with `--calib=calib/<your_file>.txt` and adjust `--stride` to suit your video frame rate and motion profile.

For multi-camera rigs, run separate sessions per camera or extend the loader to combine intrinsics appropriately.

---

## 2. Dockerized Workflow

The `docker/` directory provides an Ubuntu 22.04 + CUDA 12.1 runtime that bundles DPVO, Python dependencies, pretrained weights, and calibration files.

### 2.1 Build Locally

```bash
cd /mnt/data/pdx/DPVO/docker
docker build -t dpvo-runtime -f Dockerfile ..
```

> The build context (`..`) ensures models, calibration files, and helper scripts are copied into the image.

### 2.2 Runtime Helper (`run.sh`)

For users less familiar with Docker, the repository root exposes `run.sh`:

```bash
./run.sh /absolute/path/to/front_camera.mp4 [additional run.py args]
```

- Mounts the input video read-only in the container (`/data/input.mp4`).
- Binds the host `outputs/` directory into `/app/outputs`, so results remain immediately accessible on the host filesystem.
- Executes `python run.py` with the Tesla calibration defaults (`--calib=calib/tesla.txt --stride=5 --plot --save_camera_poses --save_motion --save_trajectory`).
- Accepts optional `run.py` flags appended to the command (e.g., `--name my_run`).
- Override defaults via environment variables:
   - `DPVO_IMAGE_NAME` – Docker image tag to use (defaults to `dpvo-runtime`).
   - `DPVO_OUTPUT_DIR` – host folder mapped to `/app/outputs` inside the container.

Treat `run.sh` as a push-button option: place your video on disk, run the script, and inspect the generated subfolder under `outputs/` when it completes.

### 2.3 Docker Hub Image *(placeholder)*

Pull the forthcoming pre-built image once it is published:

```bash
docker pull docker.io/REPO/IMAGE:TAG  # TODO: publish official image name
```

After pulling, reuse `run.sh` by exporting `DPVO_IMAGE_NAME=docker.io/REPO/IMAGE:TAG`, or invoke `docker run` manually.

### 2.4 Remote Usage on vast.ai *(placeholder)*

1. Provision a vast.ai instance with NVIDIA GPUs and Docker runtime enabled.
2. Pull the image once available:
   ```bash
   docker pull docker.io/REPO/IMAGE:TAG  # TODO: confirm tag after publication
   ```
3. Upload or mount your video on the remote host.
4. Launch DPVO inside the container:
   ```bash
   docker run --rm --gpus all --ipc=host \
     -v /abs/path/to/video.mp4:/data/input.mp4:ro \
     -v /abs/path/to/output_dir:/app/outputs \
     docker.io/REPO/IMAGE:TAG \
   python run.py --imagedir=/data/input.mp4 --calib=calib/tesla.txt --stride=5 \
       --plot --save_camera_poses --save_motion --save_trajectory
   ```

Replace the volume paths with directories available on your vast.ai instance. Saved outputs will appear directly under the mounted host directory.

### 2.5 Understanding the Output Bind Mount

- **Host side**: `outputs/` (or `DPVO_OUTPUT_DIR`) accumulates subfolders per run.
- **Container side**: DPVO writes into `/app/outputs/<run_name>`.
- Suitable for shared storage (NFS, cloud buckets mounted via FUSE) so downstream pipelines can ingest results immediately.

---

## 3. Additional Notes

- **Visualization**: Real-time visualization requires `DPViewer`. Install natively (`pip install ./DPViewer`) or extend the Docker image if you need viewer support in containers.
- **Loop Closure**: Enable SLAM back-ends with `--opts LOOP_CLOSURE True` (and optionally `--opts CLASSIC_LOOP_CLOSURE True` if classical dependencies are installed).
- **Data Refresh**: The container removes downloaded archives after extraction. Rebuild the image to pick up new checkpoints or code updates.
- **GPU Requirements**: Ensure recent NVIDIA drivers and the NVIDIA Container Toolkit are installed before running GPU-enabled containers.

---

DPVO remains an active research project. Contributions and calibration files for additional vehicle platforms are welcome.
