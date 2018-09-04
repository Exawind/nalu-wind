# -*- coding: utf-8 -*-
# pylint: disable=too-many-locals

"""
ABL Statistics plotting utility
"""

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import netCDF4 as ncdf

class ABLStatsFile(object):
    """Interface to ABL Statistics NetCDF file"""

    def __init__(self, stats_file="abl_statistics.nc"):
        """
        Args:
            stats_file (path): Absolute path to the NetCDF file
        """
        self.stats_file = stats_file
        self.abl_stats = ncdf.Dataset(stats_file)
        self._time = self.abl_stats["time"][:]
        self._heights = self.abl_stats["heights"][:]
        self.hub_height = 0.0
        self.capinv_ht1 = 0.0
        self.capinv_ht2 = 0.0

    @property
    def time(self):
        """The time array from the ABL stats file"""
        return self._time

    @property
    def heights(self):
        """Heights where data is available"""
        return self._heights

def plot_velocity(stats, pdffile, field="velocity", num_steps=3600):
    """Generate velocity plots"""
    col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ht = stats.heights
    time = stats.time
    ttmp = time[-num_steps:]
    print("Start time = %.1f s; end time = %.1f s"%(ttmp[0], ttmp[-1]))
    vel_full = stats.abl_stats.variables[field][:]
    vel = vel_full[-num_steps:, :, :]
    vmag_full = np.linalg.norm(vel, axis=-1)
    vmag_avg = np.average(vmag_full, axis=0)
    vmag_min = np.min(vmag_full, axis=0)
    vmag_max = np.max(vmag_full, axis=0)
    vavg = np.average(vel, axis=0)
    vmin = np.min(vel, axis=0)
    vmax = np.max(vel, axis=0)

    plt.figure()
    plt.plot(vmag_avg, ht, col_list[0], label=r"$|U|$")
    plt.plot(vavg[:, 0], ht, color=col_list[1], label=r"$U_x$")
    plt.plot(vavg[:, 1], ht, color=col_list[2], label=r"$U_y$")
    plt.fill_betweenx(ht, vmag_min, vmag_max, color=col_list[0], alpha=0.4)
    plt.fill_betweenx(ht, vmin[:, 0], vmax[:, 0], color=col_list[1], alpha=0.4)
    plt.fill_betweenx(ht, vmin[:, 1], vmax[:, 1], color=col_list[2], alpha=0.4)
    plt.axhline(stats.hub_height, color='k', ls='--', lw=0.5)
    plt.axhline(stats.capinv_ht1, color='k', ls='-.', lw=0.5)
    plt.axhline(stats.capinv_ht2, color='k', ls='-.', lw=0.5)
    plt.ylim(ht[0], ht[-1])
    plt.legend()
    plt.grid()
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Height (m)")
    pdffile.savefig(dpi=300, bbox_inches="tight")
    plt.close()

    wdir = 180.0 + np.degrees(np.arctan2(vavg[:, 0], vavg[:, 1]))
    plt.figure()
    plt.plot(wdir, ht)
    plt.axhline(stats.hub_height, color='k', ls='-.', lw=0.5)
    plt.axhline(stats.capinv_ht1, color='k', ls='-.', lw=0.5)
    plt.axhline(stats.capinv_ht2, color='k', ls='-.', lw=0.5)
    plt.ylim(ht[0], ht[-1])
    plt.grid()
    plt.xlabel("Wind direction (deg.)")
    plt.ylabel("Height (m)")
    pdffile.savefig(dpi=300, bbox_inches="tight")
    plt.close()

    # plt.figure()
    # plt.plot(vavg[:, 2], ht, color=col_list[3], label=r"$U_z$")
    # plt.ylim(ht[0], ht[-1])
    # plt.grid()
    # plt.xlabel(r"$U_z$ (m/s)")
    # plt.ylabel("Height (m)")
    # pdffile.savefig(dpi=300, bbox_inches="tight")
    # plt.close()

def plot_res_stress(stats, pdffile, field="resolved_stress", num_steps=3600):
    """Resolved stress"""
    col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ht = stats.heights
    idx = [0, 3, 5]
    labels = [r"$\left\langle u'u' \right \rangle$",
              r"$\left\langle v'v' \right \rangle$",
              r"$\left\langle w'w' \right \rangle$"]
    field = stats.abl_stats.variables[field][:]
    field = field[-num_steps:, :, :]
    utau = stats.abl_stats.variables["utau"][:]
    utau = np.average(utau[-num_steps:])
    utau = utau * utau
    ravg = np.average(field, axis=0) / utau
    rmin = np.min(field, axis=0) / utau
    rmax = np.max(field, axis=0) / utau

    plt.figure()
    for i, ii in enumerate(idx):
        plt.plot(ravg[:, ii], ht, col_list[i], label=labels[i])
        plt.fill_betweenx(ht, rmin[:, ii], rmax[:, ii], color=col_list[i], alpha=0.4)
    plt.axhline(stats.hub_height, color='k', ls='--', lw=0.5)
    plt.axhline(stats.capinv_ht1, color='k', ls='-.', lw=0.5)
    plt.axhline(stats.capinv_ht2, color='k', ls='-.', lw=0.5)
    plt.ylim(ht[0], ht[-1])
    plt.legend()
    plt.grid()
    plt.ylabel("Height (m)")
    plt.xlabel(r"$\left\langle u_i u_i \right\rangle/u_\tau^2$")
    pdffile.savefig(dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    labels = [r"$\left\langle u'v' \right \rangle$",
              r"$\left\langle u'w' \right \rangle$",
              r"$\left\langle v'w' \right \rangle$"]
    for i, ii in enumerate([1, 2, 4]):
        plt.plot(ravg[:, ii], ht, col_list[i], label=labels[i])
        plt.fill_betweenx(ht, rmin[:, ii], rmax[:, ii], color=col_list[i], alpha=0.4)
    plt.axhline(stats.hub_height, color='k', ls='--', lw=0.5)
    plt.axhline(stats.capinv_ht1, color='k', ls='-.', lw=0.5)
    plt.axhline(stats.capinv_ht2, color='k', ls='-.', lw=0.5)
    plt.ylim(ht[0], ht[-1])
    plt.legend()
    plt.grid()
    plt.ylabel("Height (m)")
    plt.xlabel(r"$\left\langle u_i u_j \right\rangle/u_\tau^2$")
    pdffile.savefig(dpi=300, bbox_inches="tight")
    plt.close()

def plot_phim(stats, pdffile, field="velocity", num_steps=3600):
    """Generate velocity plots"""
    ht = stats.heights
    vel_full = stats.abl_stats.variables[field][:]
    vel = vel_full[-num_steps:, :, :]
    vmag_full = np.linalg.norm(vel, axis=-1)
    vmag_avg = np.average(vmag_full, axis=0)
    utau = stats.abl_stats.variables["utau"][:]
    utau = np.average(utau[-num_steps:])
    dudz = np.gradient(vmag_avg, ht)
    phim = 0.4 / utau * ht * dudz
    plt.figure()
    plt.plot(phim, ht)
    plt.axhline(stats.hub_height, color='k', ls='--', lw=0.5)
    plt.ylim(0, 600)
    plt.xlim(0, 3)
    plt.grid()
    plt.xlabel(r"$\phi_m = {\kappa}/{u_\tau}\ \left({du}/{dz}\right) \ z$")
    plt.ylabel("Height (m)")
    pdffile.savefig(dpi=300, bbox_inches="tight")
    plt.close()

def plot_temperature(stats, pdffile, field="temperature", num_steps=3600):
    """Plot the temperature profile"""
    ht = stats.heights
    temp_full = stats.abl_stats.variables[field][:]
    temp_avg = np.average(temp_full[-num_steps:, :], axis=0)

    plt.figure()
    plt.plot(temp_avg, ht)
    plt.axhline(stats.hub_height, color='k', ls='--', lw=0.5)
    plt.axhline(stats.capinv_ht1, color='k', ls='-.', lw=0.5)
    plt.axhline(stats.capinv_ht2, color='k', ls='-.', lw=0.5)
    plt.grid()
    plt.ylim(ht[0], ht[-1])
    plt.ylabel("Height (m)")
    plt.xlabel("Temperature (K)")
    pdffile.savefig(dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    statsfile = ABLStatsFile()
    statsfile.hub_height = 70.0
    statsfile.capinv_ht1 = 650.0
    statsfile.capinv_ht2 = 750.0
    with PdfPages("owez_abl_precursor.pdf") as pdf:
        plot_velocity(statsfile, pdf)
        plot_res_stress(statsfile, pdf)
        plot_phim(statsfile, pdf)
        plot_temperature(statsfile, pdf)
