# pvpanel.py is an adaption of PyPVSim by Daniel Soto licensed under CC-BY 3.0
# https://github.com/dsoto/PyPVSim/blob/master/LICENSE.txt

from numpy import sin, cos, tan, arcsin, arccos, arctan2, nan, pi, isnan
import math
import arrow

DATE_FORMAT = 'YYYY-MM-DD HH:mm:ss'

class PVpanel:

    def __init__(self, latitude, area=1, efficiency=0.2, el_tilt=0, az_tilt=0,
                 start_date=None):
        self.area = area
        self.efficiency = efficiency
        self.el_tilt = math.radians(el_tilt)
        self.az_tilt = math.radians(az_tilt)
        self.lat = math.radians(latitude)
        if start_date is None:
            raise RuntimeError('start_date has to be given as string!')
        else:
            self.date = arrow.get(start_date, DATE_FORMAT)

    def power(self, dni):
        '''Calculate the PV panels active power output output based on the
        irradiation input and current time.'''
        p = self.area * self.efficiency * self._radiation_normal(dni)
        return p/1000000 # W -> MW

    def step_time(self, step_size):
        '''Advance the current model time'''
        self.date = self.date.shift(seconds=step_size)


    def _radiation_normal(self, dni):
        '''Calculate the normal radiation needed for the power calculation.'''
        ang = self._incidence_angle()
        if isnan(ang):
            return 0
        else:
            rn = dni * cos(self._incidence_angle())
            return max(0, rn)

    def _incidence_angle(self):
        el = self._elevation()
        az = self._azimuth()
        ang = arccos(cos(el) * cos(az - self.az_tilt) * sin(self.el_tilt)
                     + sin(el) * cos(self.el_tilt))
        return float(ang) # conversion from numpy float

    def _elevation(self):
        '''Calculate the sun's elevation at current time.'''
        dec = self._declination()
        ha = self._hour_angle()
        arg = cos(self.lat) * cos(dec) * cos(ha) + sin(self.lat) * sin(dec)
        el = arcsin(arg)
        if arg > 0:
            return el
        else:
            return nan

    def _azimuth(self):
        dec = self._declination()
        ha = self._hour_angle()
        el = self._elevation()
        # Formula from "Fundamentals of Renewable Energy Processes" (da Rosa)
        az = arctan2(sin(ha), sin(self.lat)*cos(ha) - cos(self.lat)
                     *tan(dec))
        if isnan(el):
            return nan
        else:
            return az

    def _hour_angle(self):
        arg = self.date.hour + self.date.minute / 60.0
        return math.radians(15 * (arg-12))

    def _declination(self):
        arg = 23.45 * sin(2 * pi * (self.date.day - 81) / 365.0)
        return math.radians(arg)
