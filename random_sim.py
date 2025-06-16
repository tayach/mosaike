# random_sim.py
import mosaik_api
import random

META = {
    "api_version": "3.0.6",
    "type": "time-based",
    "models": {
        "RandomData": {
            "public": True,
            "params": ["dist", "low", "high"],
            "attrs": ["value"],
            "provides": ["value"],
            "requires": [],
        },
    },
}


class RandomSim(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.entities = {}
        self.time_resolution = None

    def init(self, sid, time_resolution=60, **kwargs):
        self.time_resolution = time_resolution
        return self.meta

    def create(self, num, model, dist="uniform", low=0, high=1):
        entities = []
        for i in range(num):
            eid = f"Rand_{i}"
            self.entities[eid] = {"dist": dist, "low": low, "high": high}
            entities.append({"eid": eid, "type": model})
        return entities

    def step(self, time, inputs, max_advance=None):
        return time + self.time_resolution

    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            cfg = self.entities[eid]
            val = (
                random.uniform(cfg["low"], cfg["high"])
                if cfg["dist"] == "uniform"
                else 0
            )
            data[eid] = {"value": val}
        return data


if __name__ == "__main__":
    mosaik_api.start_simulation(RandomSim())
