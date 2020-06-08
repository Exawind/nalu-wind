#!/usr/bin/env python
#

import netCDF4 as ncdf
import numpy as np
import sys
from numpy import linalg

class ABLStatsFileClass():
    '''
    Interface to ABL Statistics NetCDF file
    '''

    def __init__(self, stats_file='abl_statistics.nc'):
        '''
        Args:
            stats_file (path): Absolute path to the NetCDF file
        '''
        # Read in the file using netcdf
        self.abl_stats = ncdf.Dataset(stats_file)

        # Extract the heights
        self.heights = self.abl_stats['heights']
    
        # Extract the time
        self.time = self.abl_stats.variables['time']

        # Extract the utau
        self.utau = self.abl_stats.variables['utau']
    
        # Velocity
        # Index - [time, height, (x, y, z)]
        velocity = self.abl_stats.variables['velocity']

        self.u   = velocity[:,:,0]
        self.v   = velocity[:,:,1]
        self.w   = velocity[:,:,2]

        # Temperature
        temperature = self.abl_stats.variables['temperature']

    def get_latesttime(self, field='velocity', index=0, scalar=False, zeroD=False):
        '''
        Provide field time average
        field - the field to time-average
        index - the component index (for example: velocity has 3 components)
        '''
        
        # Filter the field based on the times
        filt = -1 #((self.time[:] >= times[0]) & (self.time[:] <= times[1]))
        # Filtered time
        t = self.time[filt]

        # Filtered field
        if zeroD:         f = self.abl_stats[field][filt]
        elif scalar:      f = self.abl_stats[field][filt,:]
        else:             f = self.abl_stats[field][filt,:,index]

        return f

    def time_average(self, field='velocity', index=0, times=[0., 100], scalar=False, zeroD=False):
        '''
        Provide field time average
        field - the field to time-average
        index - the component index (for example: velocity has 3 components)
        times - the times to average
        '''
        
        # Filter the field based on the times
        filt = ((self.time[:] >= times[0]) & (self.time[:] <= times[1]))
        # Filtered time
        t = self.time[filt]
        # The total time
        dt = np.amax(t) - np.amin(t)

        # Filtered field
        if zeroD:         f = self.abl_stats[field][filt]
        elif scalar:      f = self.abl_stats[field][filt,:]
        else:             f = self.abl_stats[field][filt,:,index]

        # Compute the time average as an integral
        integral = np.trapz(f, x=t, axis=0) / dt

        return integral

def calcdelta(tavgfield, lasttimefield, indexi, times):
    """Get the difference between two fields, tavgfield and
    lasttimefield, at indexi
    """
    q1 = data.time_average(field=tavgfield, index=indexi, times=times)
    q2 = data.get_latesttime(field=lasttimefield, index=indexi) 
    return linalg.norm(q1-q2)

# Lambda to report a quick PASS/FAIL
pf = lambda x, tol : 'PASS' if x < tol else 'FAIL'

# Get the argument
ablnc_filename         ='abl_statistics.nc'
resolvedstress_filename='abl_resolved_stress_stats.dat'

Tfluxtol = 5.0E-8
SFStol   = 5.0E-2
TKEtol   = 1.0E-3
passtest = True

# Load the data
data = ABLStatsFileClass(stats_file=ablnc_filename)
time = data.time
t1   = min(time)
t2   = max(time)

# --- Temperature flux ---
dflux0 = calcdelta('temperature_resolved_flux', 
                   'temperature_resolved_flux_tavg', 0, [t1, t2])
dflux1 = calcdelta('temperature_resolved_flux', 
                   'temperature_resolved_flux_tavg', 1, [t1, t2])
dflux2 = calcdelta('temperature_resolved_flux', 
                   'temperature_resolved_flux_tavg', 2, [t1, t2])

if all(x<Tfluxtol for x in [dflux0, dflux1, dflux2])==False:
    passtest = False
    
print("<Tu> diff L2:   %e TOL: %e %s"%(dflux0, Tfluxtol, pf(dflux0, Tfluxtol)))
print("<Tv> diff L2:   %e TOL: %e %s"%(dflux1, Tfluxtol, pf(dflux0, Tfluxtol)))
print("<Tw> diff L2:   %e TOL: %e %s"%(dflux2, Tfluxtol, pf(dflux0, Tfluxtol)))
# ------------------------------

# --- SFS stress ---
dsfs0  = calcdelta('sfs_stress', 'sfs_stress_tavg', 0, [t1, t2])
dsfs3  = calcdelta('sfs_stress', 'sfs_stress_tavg', 3, [t1, t2])
dsfs5  = calcdelta('sfs_stress', 'sfs_stress_tavg', 5, [t1, t2])
print("<SFSuu> diff L2: %e TOL: %e %s"%(dsfs0, SFStol, pf(dsfs0, SFStol)))
print("<SFSvv> diff L2: %e TOL: %e %s"%(dsfs3, SFStol, pf(dsfs0, SFStol)))
print("<SFSww> diff L2: %e TOL: %e %s"%(dsfs5, SFStol, pf(dsfs0, SFStol)))
if all(x<SFStol for x in [dsfs0, dsfs3, dsfs5])==False:
    passtest = False
# -----------------

# --- Resolved TKE ----
fieldstr='resolved_stress'
uu      = data.time_average(field=fieldstr, index=0, times=[t1, t2])
vv      = data.time_average(field=fieldstr, index=3, times=[t1, t2])
ww      = data.time_average(field=fieldstr, index=5, times=[t1, t2])
TKEdat  = np.loadtxt(resolvedstress_filename)
duu     = linalg.norm(uu - TKEdat[:,1])
dvv     = linalg.norm(vv - TKEdat[:,4])
dww     = linalg.norm(ww - TKEdat[:,6])
print("<uu> diff L2: %e TOL: %e %s"%(duu, TKEtol, pf(duu, TKEtol)))
print("<vv> diff L2: %e TOL: %e %s"%(dvv, TKEtol, pf(duu, TKEtol)))
print("<ww> diff L2: %e TOL: %e %s"%(dww, TKEtol, pf(duu, TKEtol)))
if all(x<TKEtol for x in [duu, dvv, dww])==False:
    passtest = False

# -------------

if passtest:
    sys.exit(0)
else:
    sys.exit(1)
