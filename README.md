PIK3CA mutation detection
==============================

Implementation Additive MIL model for pik3ca mutation detection in histopathological images

-------
To set up the environment and avoid path issues, run:
```
pip install -e .
```

To launch a model, use the following command:
```
python main.py --model AdditiveMIL
```
-------
Repository structure:
```
.
├── data/           # Data directory
├── models/         # Model architectures
├── utils/          # Utility scripts and helper functions
├── results/        # Output and results from model runs
├── notebooks/      # Jupyter notebooks
├── main.py         # Main script to run models
├── prediction.py   # Run a saved model on the testset
├── main_wandb.py   # Script used to do hyperparams sweeping
└── README.md       # This document
```

