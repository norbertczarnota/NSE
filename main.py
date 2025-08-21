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
def avg_u_to_center(u):
    return 0.5 * (u[0:-1, :, :] + u[1:, :, :])

def avg_v_to_center(v):
    return 0.5 * (v[:, 0:-1, :] + v[:, 1:, :])

def avg_w_to_center(w):
    return 0.5 * (w[:, :, 0:-1] + w[:, :, 1:])

def avg_center_to_u(u_center):
    Nx, Ny, Nz = u_center.shape
    u_face = np.empty((Nx+1, Ny, Nz), dtype=u_center.dtype)
    u_face[1:-1, :, :] = 0.5 * (u_center[0:-1, :, :] + u_center[1:, :, :])
    u_face[0, :, :] = 0.5 * (u_center[-1, :, :] + u_center[0, :, :])
    u_face[-1, :, :] = u_face[0, :, :]
    return u_face

def avg_center_to_v(v_center):
    Nx, Ny, Nz = v_center.shape
    v_face = np.empty((Nx, Ny+1, Nz), dtype=v_center.dtype)
    v_face[:,1:-1,:] = 0.5 * (v_center[:,0:-1,:] + v_center[:,1:,:])
    v_face[:,0,:] = 0.5 * (v_center[:,-1,:] + v_center[:,0,:])
    v_face[:,-1,:] = v_face[:,0,:]
    return v_face

def avg_center_to_w(w_center):
    Nx, Ny, Nz = w_center.shape
    w_face = np.empty((Nx, Ny, Nz+1), dtype=w_center.dtype)
    w_face[:,:,1:-1] = 0.5 * (w_center[:,:,0:-1] + w_center[:,:,1:])
    w_face[:,:,0] = 0.5 * (w_center[:,:,-1] + w_center[:,:,0])
    w_face[:,:,-1] = w_face[:,:,0]
    return w_face

def laplacian(arr, dx, dy, dz):
    return ((np.roll(arr, -1, axis=0) - 2*arr + np.roll(arr, 1, axis=0))/dx**2 +
            (np.roll(arr, -1, axis=1) - 2*arr + np.roll(arr, 1, axis=1))/dy**2 +
            (np.roll(arr, -1, axis=2) - 2*arr + np.roll(arr, 1, axis=2))/dz**2)

def divergence_mac(u, v, w, dx, dy, dz):
    div = (u[1:,:,:] - u[:-1,:,:]) / dx
    div += (v[:,1:,:] - v[:,:-1,:]) / dy
    div += (w[:,:,1:] - w[:,:,:-1]) / dz
    return div

def grad_phi_to_faces(phi, dx, dy, dz): #idk if this works properly
    Nx, Ny, Nz = phi.shape
    grad_x = np.empty((Nx+1, Ny, Nz), dtype=phi.dtype)
    grad_x[1:-1,:,:] = (phi[1:,:,:] - phi[:-1,:,:]) / dx
    grad_x[0,:,:] = (phi[0,:,:] - phi[-1,:,:]) / dx
    grad_x[-1,:,:] = grad_x[0,:,:]
    grad_y = np.empty((Nx, Ny+1, Nz), dtype=phi.dtype)
    grad_y[:,1:-1,:] = (phi[:,1:,:] - phi[:,:-1,:]) / dy
    grad_y[:,0,:] = (phi[:,0,:] - phi[:,-1,:]) / dy
    grad_y[:,-1,:] = grad_y[:,0,:]
    grad_z = np.empty((Nx, Ny, Nz+1), dtype=phi.dtype)
    grad_z[:,:,1:-1] = (phi[:,:,1:] - phi[:,:,:-1]) / dz
    grad_z[:,:,0] = (phi[:,:,0] - phi[:,:,-1]) / dz
    grad_z[:,:,-1] = grad_z[:,:,0]
    return grad_x, grad_y, grad_z

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