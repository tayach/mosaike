import pandas as pd
import numpy as np

import mosaik_api_v3 as mosaik_api

DATE_FORMAT = 'YYYY-MM-DD HH:mm:ss'

SENTINEL = object()


class CSV(mosaik_api.Simulator):
    def __init__(self):
        super().__init__({'models': {}})
        self.start_date = None
        self.data = None
        self.attrs = None
        self.cache = None
        self.sid = None
        self.eid = None
        self.eids = []
        self.delimiter = None
        self.type = None
        self.time_res = None
        self.next_date = None
        self.model_name = None
        self.next_index = None
        
    def init(self, sid, time_resolution, sim_start, datafile, date_format=None, type="time-based", delimiter=','):
        self.type = type
        if self.type != "time-based" and self.type != "event-based":
            print("Please enter the correct type. The type can either be time-based or event-based")
        self.sid = sid
        self.time_res = pd.Timedelta(time_resolution, unit='seconds')
        start_date = self.start_date = pd.to_datetime(sim_start, format=date_format)
        self.next_date = self.start_date
        self.delimiter = delimiter

        # Check if first line is the header with column names (our attributes)
        # or a model name:
        with open(datafile) as f:
            first_line = f.readline()
        first_line = first_line.strip('\n')

        if len(first_line.split(self.delimiter)) == 1:
            self.model_name = first_line
            header = 1
        else:
            header = 0
            self.model_name = 'Data'

        data = self.data = pd.read_csv(datafile, index_col=0, parse_dates=True,
                                       header=header)
        data.rename(columns=lambda x: x.strip(), inplace=True)

        self.attrs = [attr.strip() for attr in data.columns]

        for i, attr in enumerate(self.attrs):
            try:
                # Try stripping comments
                attr = attr[:attr.index('#')]
            except ValueError:
                pass
            self.attrs[i] = attr.strip()

        # Rename column names of datafrmae
        data.columns = self.attrs

        self.meta['type'] = self.type

        self.meta['models'][self.model_name] = {
            'public': True,
            'params': [],
            'attrs': self.attrs,
        }

        # Find first relevant value:
        if self.type == "time-based":
            first_index = data.index.get_indexer([start_date], method='ffill')[0]
            self.next_index = first_index
        else:
            first_index = data.index.get_indexer([start_date], method='bfill')[0]
            first_date = data.index[first_index]
            if first_date == start_date:
                self.next_index = first_index
            else:
                self.next_index = -1

        return self.meta

    def create(self, num, model):
        if model != self.model_name:
            raise ValueError('Invalid model "%s" % model')

        start_idx = len(self.eids)
        entities = []
        for i in range(num):
            eid = '%s_%s' % (model, i + start_idx)
            entities.append({
                'eid': eid,
                'type': model,
                'rel': [],
            })
            self.eids.append(eid)
        return entities

    def step(self, time, inputs, max_advance):
        data = self.data
        if self.next_index >= 0:
            current_row = data.iloc[self.next_index]
            self.cache = dict(current_row)
        else:
            self.cache = {}
        self.next_index += 1
        try:
            next_date = self.data.index[self.next_index]
            next_step = int((next_date - self.start_date)/self.time_res)
        except IndexError:
            next_step = max_advance

        return next_step

    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            if eid not in self.eids:
                raise ValueError('Unknown entity ID "%s"' % eid)
            data[eid] = {}
            for attr in attrs:
                if isinstance(self.cache[attr], np.floating) or isinstance(self.cache[attr], np.integer):
                    data[eid][attr] = self.cache[attr].tolist()
                else:
                    data[eid][attr] = str(self.cache[attr])  

        return data


def main():
    return mosaik_api.start_simulation(CSV(), 'mosaik-csv simulator')


if __name__ == '__main__':
    main()
