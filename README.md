# Mosaik Demo

This repository contains a simple Mosaik setup with a grid simulator,
a random data generator, and an HDF5 logger. Use the following steps to
run the example scenario.

## 1. Install dependencies

Create a virtual environment (optional) and install the required Python
packages:

```bash
pip install -r requirements.txt
```

## 2. Generate the grid description

Run the helper script to generate `grid.json` which contains the grid
model used by the simulators:

```bash
python generate_grid.py
```

## 3. Prepare load profiles

The scenario expects a file named `load_profiles.data` with load data
for the grid. Create this file or obtain it from your data source and
place it in the repository root.

## 4. Run the simulation

Execute the scenario which wires the simulators together and steps
through the time horizon:

```bash
python scenario.py
```

## 5. Inspect results

After the run finishes the results are written to `results.hdf5`. No
`results.csv` file is generated. The HDF5 file contains the collected
outputs of the grid simulation and can be analysed with your preferred
tools.
