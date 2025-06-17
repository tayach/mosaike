# csv_writer.py
import mosaik_api
import csv

META = {
    "api_version": "3.0.6",
    "type": "time-based",
    "models": {
        "CSVWriter": {
            "public": True,
            "params": ["output_file"],
            "attrs": ["vm_pu"],
            "requires": ["vm_pu"],
        },
    },
}


class CSVWriter(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.file = None
        self.writer = None
        self.header_written = False
        self.time_resolution = None

    def init(self, sid, time_resolution=60, output_file="results.csv", **kwargs):
        self.time_resolution = int(time_resolution)
        self.file = open(output_file, "w", newline="")
        self.writer = csv.writer(self.file)
        return self.meta

    def create(self, num, model):
        return [{"eid": f"Writer_{i}", "type": model} for i in range(num)]

    def step(self, time, inputs, max_advance=None):
        # ``inputs`` has the structure {writer_eid: {Bus_X: {'vm_pu': val}, ...}}
        # We only create a single writer entity, so take the first mapping and
        # write one column per bus.
        if not inputs:
            return time + self.time_resolution

        writer_data = next(iter(inputs.values()))
        vm_pu_map = writer_data.get("vm_pu", {})
        buses = sorted(vm_pu_map.keys())

        if not self.header_written:
            header = ["time"] + buses
            self.writer.writerow(header)
            self.header_written = True

        row = [time]
        for bus in buses:
            val = vm_pu_map.get(bus)
            row.append(val)

        self.writer.writerow(row)
        self.file.flush()
        return time + self.time_resolution

    def get_data(self, outputs):
        return {}

    def finalize(self):
        if self.file:
            self.file.close()


if __name__ == "__main__":
    mosaik_api.start_simulation(CSVWriter())
