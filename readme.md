## Setup Instructions

### 1. Create and Activate a Virtual Environment

```bash
python3 -m venv gen-env
source gen-env/bin/activate
```

### 2. Install Dependencies

Make sure you're inside the virtual environment and run:

```bash
pip install -r requirements.txt
```


## 📁 Project Structure

```
├── celebA                     # Root dataset directory
│   └── celeba                 # Actual CelebA dataset location
├── celebA-GAN.ipynb           # GAN notebook
├── celebA-vAE.ipynb           # Variational Autoencoder notebook
├── evaluators                 # Evaluation notebooks
│   ├── classifier.ipynb
│   ├── fid.ipynb
│   └── t-SNE.ipynb
├── logs                       # Runtime logs and outputs
│   ├── error.log
│   ├── output.log
│   └── output.txt
├── scripts                    # Auto-generated notebooks with timestamped names
│   └── celebA-GAN-*.ipynb
├── vae_outputs                # Generated images from VAE
├── gan_outputs               # Generated images from GAN
├── run_notebook.csh           # SLURM batch job script
├── run_notebook.py            # Python script to execute notebooks with papermill
├── requirements.txt           # Python dependencies
└── readme.md                  # Project documentation
```

## Evaluation

Evaluate the quality of generated images using the notebooks in the `evaluators/` directory:
- `classifier.ipynb` – Accuracy of image classification
- `fid.ipynb` – Frechet Inception Distance
- `t-SNE.ipynb` – t-SNE projection for visual inspection

## Notes

- Dataset can be downloaded from `https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg`
- Generated notebooks are saved in the `scripts/` folder with timestamps.
- Log files are written to the `logs/` folder.
- You can switch between model runs by commenting/uncommenting lines in `run_notebook.py`.
- You can run using batch command `sbatch run_notebook.csh`
- Make sure the CelebA dataset is located at `celebA/celeba/`.