import os
import sys
import time
import math
import argparse
import itertools
import numpy as np
from scipy.fft import rfftn, irfftn, fftn, ifftn, fftfreq, rfftfreq
from datetime import datetime

#variables
def default_config():
    return {
        'Nx': 64, 'Ny': 64, 'Nz': 64,
        'Lx': 2*np.pi, 'Ly': 2*np.pi, 'Lz': 2*np.pi,
        'Re': 100.0,
        'IC': 'taylor-green',
        'IC_params': {},
        'CFL': 0.5,
        'safety': 0.5,
        't_end': 1.0,
        'max_steps': 1000,
        'dt_max': 1e-3,
        'output_interval': 50,
        'forcing': False,
        'forcing_params': {},
        'RNG_seed': 12345,
        'save_prefix': 'run',
        'blowup_enable': True,
        'bkm_alert_threshold': 1e-1,
        'omega_growth_factor': 5.0,
        'omega_min_growth_rate': 1e-3,
        'spectral_pileup_ratio': 1e-3,
        'blowup_check_window': 10,
        'blowup_local_patch_radius': 8,
        'blowup_extra_output_steps': 50,
        'blowup_crash_dump_dir': 'crash_dumps',
        'blowup_verbose': True,
        # Thresholds (come back to this they're definitley wrong)
        'blowup_enable': True,
        'bkm_alert_threshold': 1e-1,
        'omega_growth_factor': 5.0,
        'omega_min_growth_rate': 1e-3,
        'spectral_pileup_ratio': 1e-3,
        'blowup_check_window': 10,
        'bkm_delta_threshold': 1e-2,
        'blowup_fit_r2': 0.90,
        'blowup_fit_alpha': 0.5,
        'blowup_fit_time_horizon_factor': 5.0,
        'blowup_min_consecutive_triggers': 2,
    }

#grid util
def create_grid(Nx, Ny, Nz, Lx, Ly, Lz):
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    kx = 2*np.pi * fftfreq(Nx, d=dx)
    ky = 2*np.pi * fftfreq(Ny, d=dy)
    kz = 2*np.pi * rfftfreq(Nz, d=dz)
    return {'dx': dx, 'dy': dy, 'dz': dz, 'kx': kx, 'ky': ky, 'kz': kz, 'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'Lx': Lx, 'Ly': Ly, 'Lz': Lz}

#interpolations and MAC operators

#convective term? leave untill the end

#poisson solver

#forcing util

#diagnostics

#initial conditions

#blowup detectors

#I/O helpers

#time stepper & main

#batch sweeper

#parser

#entrypoint