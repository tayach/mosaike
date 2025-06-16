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
            "attrs": [],  # no internal state to expose
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
        self.time_resolution = time_resolution
        self.file = open(output_file, "w", newline="")
        self.writer = csv.writer(self.file)
        return self.meta

    def create(self, num, model):
        return [{"eid": f"Writer_{i}", "type": model} for i in range(num)]

    def step(self, time, inputs, max_advance=None):
        # inputs: {eid: {Bus_X: {'vm_pu': 1.0}, ...}, ...}
        if inputs:
            # Flatten into one row per time
            row = [time]
            for eid, data in inputs.items():
                # extract numeric value from nested mapping
                if data:
                    attr_map = next(iter(data.values()))
                    val = next(iter(attr_map.values()))
                    row.append(val)
            if not self.header_written:
                header = ["time"] + list(inputs.keys())
                self.writer.writerow(header)
                self.header_written = True
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
