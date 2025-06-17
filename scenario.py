import mosaik

SIM_CONFIG = {
    "GridSim": {
        "python": "grid:PandapowerSim",
    },
}

END = 3600  # 1 hour simulation
STEP_SIZE = 60  # 1-minute time steps


def main():
    world = mosaik.World(SIM_CONFIG)

    # Start simulator
    grid_sim = world.start("GridSim", step_size=STEP_SIZE)

    # Create grid entities (e.g., 5 buses)
    grid_sim.OberrheinGrid.create(5)

    # Run the simulation
    world.run(until=END)


if __name__ == "__main__":
    main()
