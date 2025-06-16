import mosaik_api
import pandapower.networks as nw
import pandapower as pp

META = {
    "api_version": "3.0.6",
    "type": "time-based",
    "models": {
        "OberrheinGrid": {
            "public": True,
            "params": [],
            "attrs": ["vm_pu", "p_mw"],
            "provides": ["vm_pu"],
            "requires": ["p_mw"],
        },
    },
}


class PandapowerSim(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid_prefix = "Bus_"
        self.entities = {}
        self.time = 0
        self.net = nw.mv_oberrhein(scenario="load")

    def init(self, sid, time_resolution=60, **kwargs):
        self.time_resolution = int(time_resolution)
        return self.meta

    def create(self, num, model):
        entities = []
        for i, bus in enumerate(self.net.bus.index[:num]):
            eid = f"{self.eid_prefix}{bus}"
            self.entities[eid] = {"bus": bus}
            entities.append({"eid": eid, "type": model})
        return entities

    def step(self, time, inputs, max_advance=None):
        for eid, sources in inputs.items():
            if sources:
                attr_map = next(iter(sources.values()))
                if "p_mw" in attr_map:
                    bus = self.entities[eid]["bus"]
                    self.net.load.loc[
                        self.net.load.bus == bus, "p_mw"
                    ] = attr_map["p_mw"]
        pp.runpp(self.net)
        self.time += self.time_resolution
        return self.time

    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            bus = self.entities[eid]["bus"]
            vm_pu = self.net.res_bus.vm_pu.at[bus]
            data[eid] = {"vm_pu": vm_pu}
        return data


if __name__ == "__main__":
    mosaik_api.start_simulation(PandapowerSim())
