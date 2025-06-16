import mosaik

SIM_CONFIG = {
    "GridSim": {
        "python": "grid:PandapowerSim",
    },
    "Random": {
        "python": "random_sim:RandomSim",
    },
    "CSVWriter": {
        "python": "csv_writer:CSVWriter",
    },
}

END = 3600  # 1 hour simulation
STEP_SIZE = 60  # 1-minute time steps


def main():
    world = mosaik.World(SIM_CONFIG)

    # Start simulators
    grid_sim = world.start("GridSim", step_size=STEP_SIZE)
    rand_sim = world.start("Random", step_size=STEP_SIZE)
    csv_writer = world.start(
        "CSVWriter", step_size=STEP_SIZE, output_file="results.csv"
    )

    # Create grid entities (e.g., 5 buses)
    grid_entities = grid_sim.OberrheinGrid.create(5)

    # Create random sources for each bus
    random_sources = rand_sim.RandomData.create(5, dist="uniform", low=0, high=5)

    # Connect random source to grid inputs
    for rand, grid in zip(random_sources, grid_entities):
        world.connect(rand, grid, "value", "p_mw")

    # Connect grid outputs to CSV logger
    for grid in grid_entities:
        world.connect(grid, csv_writer, "vm_pu")

    # Run the simulation
    world.run(until=END)


if __name__ == "__main__":
    main()
