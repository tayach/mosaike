import itertools
import mosaik_api_v3

from mosaik_components.pv.pvpanel import PVpanel

meta = {
    'models': {
        'PV': {
            'public': True,
            'params': [
                'latitude',     # latitude of data measurement location [°]
                'area',         # area of panel [m2]
                'efficiency',   # panel efficiency
                'el_tilt',      # panel elevation tilt [°]
                'az_tilt',      # panel azimuth tilt [°]
            ],
            'attrs': ['P[MW]',      # output active power [MW]
                      'DNI[W/m2]',    # input direct normal insolation [W/m2]
                      'scale_factor']    # input of modifier from ctrl
        },
    },
}

DATE_FORMAT = 'YYYY-MM-DD HH:mm:ss'

class PVSimulator(mosaik_api_v3.Simulator):
    def __init__(self):
        super(PVSimulator, self).__init__(meta)
        self.sid = None

        self.gen_neg = True     # true if generation is negative
        self.cache = None

        self._entities = {}
        self.eid_counters = {}

    def init(self, sid, start_date, step_size, gen_neg=False, time_resolution=1):
        self.sid = sid
        self.gen_neg = gen_neg

        self.start_date = start_date
        self.step_size = step_size

        return self.meta

    def create(self, num, model, **model_params):
        counter = self.eid_counters.setdefault(model, itertools.count()) #This is the Value to be returned in case model is not found.

        entities = []

        # creation of the entities:
        for i in range(num):
            eid = '%s_%s' % (model, next(counter))

            self._entities[eid] = PVpanel(start_date=self.start_date,
                                                  **model_params)

            entities.append({'eid': eid, 'type': model, 'rel': []})

        return entities

    def step(self, time, inputs, max_advance=3600):
        self.cache = {}
        for eid, attrs in inputs.items():
            scale_factor = 1
            for attr, vals in attrs.items():
                if attr == 'DNI[W/m2]':
                    dni = list(vals.values())[0] # only one source expected
                    self.cache[eid] = self._entities[eid].power(dni)
                    self._entities[eid].step_time(self.step_size)
                    if self.gen_neg:
                        self.cache[eid] *= (-1)
                elif attr == 'scale_factor':
                    scale_factor = list(vals.values())[0]

            self.cache[eid] *= scale_factor


        return time + self.step_size

    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            if eid not in self._entities.keys():
                raise ValueError('Unknown entity ID "%s"' % eid)

            data[eid] = {}
            for attr in attrs:
                if attr != 'P[MW]':
                    raise ValueError('Unknown output attribute "%s"' % attr)
                data[eid][attr] = self.cache[eid]
        return data


def main():
    mosaik_api_v3.start_simulation(PVSimulator(), 'PV-Simulator')
