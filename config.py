from dataclasses import dataclass

@dataclass
class Config:
    base_path: str
    hpc_run: bool
    synth_data: bool
    synth_data_stamp_name: str
    num_excitatory_cells: int
    num_epochs: int

# Sample configuration for running on an HPC cluster
HPC_CONFIG = Config(
    base_path="/rds/general/user/rhc222/home/Thesis/",
    hpc_run=True,
    synth_data=False,
    synth_data_stamp_name="20230810_140641",
    num_excitatory_cells=1000,
    num_epochs=30,
)

# Sample configuration for running locally
LOCAL_CONFIG = Config(
    base_path=r"C:\\Users\\username\\OneDrive - Imperial College London\\0_Imperial_main_asof_1.19.23\\0Thesis_Project\\0MAIN",
    hpc_run=False,
    synth_data=True,
    synth_data_stamp_name="20230809_175500",
    num_excitatory_cells=10,
    num_epochs=10,
)

# Select which configuration to use based on environment variable
import os

# Set the environment variable ``GAT_CONFIG`` to ``"HPC"`` to use ``HPC_CONFIG``.
# Any other value (or if the variable is unset) will fall back to ``LOCAL_CONFIG``.
use_hpc = os.getenv("GAT_CONFIG", "LOCAL").upper() == "HPC"

CONFIG = HPC_CONFIG if use_hpc else LOCAL_CONFIG
