import mosaik

SIM_CONFIG = {
    "GridSim": {
        "python": "mosaik_pandapower.simulator:Pandapower",
    },
    "HouseholdSim": {
        "python": "mosaik_householdsim.mosaik:HouseholdSim",
    },
    "HDF5": {
        "cmd": "mosaik-hdf5 %(addr)s",
    },
}

END = 3600  # 1 hour simulation
STEP_SIZE = 900  # 15-minute time steps


def main():
    world = mosaik.World(SIM_CONFIG)

    # Start simulators
    grid_sim = world.start("GridSim", step_size=STEP_SIZE)
    house_sim = world.start("HouseholdSim", step_size=STEP_SIZE)
    hdf5 = world.start("HDF5", step_size=STEP_SIZE, duration=END)

    # Load grid and residential profiles
    grid = grid_sim.Grid.create(
        1, gridfile="grid.json", sim_start="2025-01-01 00:00:00"
    )[0]
    resid = house_sim.ResidentialLoads.create(
        1,
        sim_start="2025-01-01 00:00:00",
        profile_file="load_profiles.data",
        grid_name="grid",
    )[0]
    houses = resid.children

    # Map buses by their ID without the simulator prefix
    buses = [c for c in grid.children if c.type == "Bus"]
    bus_map = {b.eid.split("-", 1)[1]: b for b in buses}

    # Connect houses to their load buses
    node_ids = world.get_data(houses, "node_id")
    for house in houses:
        node_id = node_ids[house.eid]["node_id"]
        bus = bus_map.get(node_id)
        if bus:
            world.connect(house, bus, ("P_out", "p_mw"))

    # Create database entity
    db = hdf5.Database.create(1, filename="results.hdf5")[0]

    # Connect bus results to database
    for bus in buses:
        world.connect(bus, db, "vm_pu")

    # Run the simulation
    world.run(until=END)


if __name__ == "__main__":
    main()
