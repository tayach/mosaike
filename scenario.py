import mosaik
import csv_writer


SIM_CONFIG = {
    "GridSim": {
        "python": "mosaik_pandapower.simulator:Pandapower",
    },
    "HouseholdSim": {
        "python": "householdsim.mosaik:HouseholdSim",
    },
    "CSVWriter": {
        "python": "csv_writer:CSVWriter",
    },
}


END = 3600  # 1 hour simulation
STEP_SIZE = 900  # 15-minute time steps


def main():
    world = mosaik.World(SIM_CONFIG)

    grid_sim = world.start(
        "GridSim", time_resolution=STEP_SIZE, step_size=STEP_SIZE
    )
    hh_sim = world.start("HouseholdSim", time_resolution=1)
    csv_writer = world.start(
        "CSVWriter", step_size=STEP_SIZE, output_file="results.csv"
    )

    # Load grid description
    grid = grid_sim.Grid.create(1, gridfile="grid.json")[0]

    # Create residential load entities
    houses_parent = hh_sim.ResidentialLoads.create(
        1,
        sim_start="2014-01-01 00:00:00",
        profile_file="load_profiles.data",
        grid_name="Oberrhein",
    )[0]
    houses = houses_parent.children

    # Connect houses to their buses
    node_ids = world.get_data(houses, "node_id")
    buses = {bus.eid: bus for bus in grid.children if bus.type == "Bus"}
    for house in houses:
        node_id = node_ids[house]["node_id"]
        bus = buses.get(node_id)
        if bus:
            world.connect(house, bus, ("P_out", "p_mw"))

    # Store bus results in CSV file
    writer = csv_writer.CSVWriter.create(1)[0]
    for bus in buses.values():
        world.connect(bus, writer, "vm_pu")

    world.run(until=END)


if __name__ == "__main__":
    main()
