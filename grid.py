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
        # Use buses that actually have loads so that varying the ``p_mw``
        # values influences the power flow results. ``net.load.bus`` contains
        # the bus indices for all loads in the network. Select the first
        # ``num`` unique bus indices from this list.
        load_buses = list(dict.fromkeys(self.net.load.bus.values))[:num]
        for bus in load_buses:
            eid = f"{self.eid_prefix}{bus}"
            self.entities[eid] = {"bus": bus}
            entities.append({"eid": eid, "type": model})
        return entities

    def step(self, time, inputs, max_advance=None):
        for eid, sources in inputs.items():
            if sources:
                # ``sources`` maps src_eid -> {attr_name: value}. We only
                # expect a single source attribute per bus, so extract the
                # numeric value regardless of the attribute name.
                attr_map = next(iter(sources.values()))
                if attr_map:
                    val = next(iter(attr_map.values()))
                    bus = self.entities[eid]["bus"]
                    self.net.load.loc[
                        self.net.load.bus == bus, "p_mw"
                    ] = val
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
