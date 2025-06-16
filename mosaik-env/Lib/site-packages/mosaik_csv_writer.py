"""
A simple data collector that saves all input into a csv file.

"""
import pandas as pd
import mosaik_api_v3 as mosaik_api
import numpy as np

META = {
    'type': 'event-based',
    'models': {
        'CSVWriter': {
            'public': True,
            'any_inputs': True,
            'params': ['buff_size', 'attrs'],
            'attrs': [],
        },
    },
}


class CSVWriter(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid = None
        self.time_resolution = None
        self.date_format = None
        self.start_date = None
        self.output_file = None
        self.df = None
        self.buff_size = None
        self.attrs = None
        self.nan_representation = None

    def init(self, sid, time_resolution, start_date,
             date_format='%Y-%m-%d %H:%M:%S', output_file='results.csv',
             nan_rep='NaN'):
        self.time_resolution = time_resolution
        self.date_format = date_format
        self.start_date = pd.to_datetime(start_date, format=date_format)
        self.output_file = output_file
        self.nan_representation = nan_rep
        self.df = pd.DataFrame([])

        return self.meta

    def create(self, num, model, buff_size=500, attrs=None):
        if num > 1 or self.eid is not None:
            raise RuntimeError('Can only create one instance of CSVWriter.')
        if attrs:
            self.attrs = ['date']
            self.attrs.extend(attrs)
        self.buff_size = buff_size
        self.eid = 'CSVWriter'

        return [{'eid': self.eid, 'type': model}]

    def step(self, time, inputs, max_advance):
        current_date = (self.start_date
                        + pd.Timedelta(time * self.time_resolution, unit='seconds'))

        data_dict = {'date': current_date}
        for attr, values in inputs.get(self.eid, {}).items():
            for src, value in values.items():
                data_dict[f'{src}-{attr}'] = [value]
        if self.attrs:
            # List of attributes was provided in create() function or created in first step
            # Based on this list the DataFrame is created. Thus, also attributes with no input
            # are in the Dataframe as NaN values.
            df_data = pd.DataFrame(data_dict, columns=self.attrs)
            df_data.set_index('date', inplace=True)
        else:
            # Initialize attribute list based on the attributes in the first step
            # This might be problematic, if in a later step more attributes are provided
            self.attrs = list(data_dict.keys())
            df_data = pd.DataFrame(data_dict)
            df_data.set_index('date', inplace=True)

        if time == 0:
            self.df = df_data
            self.df.to_csv(self.output_file,
                           mode='w',
                           header=True,
                           date_format=self.date_format,
                           na_rep=self.nan_representation)
            self.df = pd.DataFrame([])
        elif time > 0:
            self.df = pd.concat([self.df, df_data])
            if time % self.buff_size == 0:
                self.df.to_csv(self.output_file,
                               mode='a',
                               header=False,
                               date_format=self.date_format,
                               na_rep=self.nan_representation)
                self.df = pd.DataFrame([])

        return None

    def finalize(self):
        self.df.to_csv(self.output_file,
                       mode='a',
                       header=False,
                       date_format=self.date_format,
                       na_rep=self.nan_representation)


if __name__ == '__main__':
    mosaik_api.start_simulation(CSVWriter())
