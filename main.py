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
def solve_poisson_fft(rhs, grid):
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    R_hat = rfftn(rhs, s=(Nx, Ny, Nz))
    kx = grid['kx'][:, None, None]
    ky = grid['ky'][None, :, None]
    kz = grid['kz'][None, None, :]
    k2 = kx**2 + ky**2 + kz**2
    phi_hat = np.zeros_like(R_hat)
    mask = (k2 != 0.0)
    phi_hat[mask] = R_hat[mask] / (-k2[mask])
    phi_hat[0,0,0] = 0.0
    phi = irfftn(phi_hat, s=(Nx, Ny, Nz))
    return phi


#forcing util
def compute_forcing_fields(grid, kind=None, params=None):
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
    Lx, Ly, Lz = grid['Lx'], grid['Ly'], grid['Lz']
    if not kind:
        return np.zeros((Nx+1, Ny, Nz)), np.zeros((Nx, Ny+1, Nz)), np.zeros((Nx, Ny, Nz+1))
    if kind == 'kolmogorov':
        F = params.get('F', 1.0); k = params.get('k', 1)
        y_u = (np.arange(Ny) + 0.5) * dy
        f_u = F * np.sin(k * y_u)[None, :, None]
        f_u = np.tile(f_u, (Nx+1, 1, Nz))
        return f_u, np.zeros((Nx, Ny+1, Nz)), np.zeros((Nx, Ny, Nz+1))
    return np.zeros((Nx+1, Ny, Nz)), np.zeros((Nx, Ny+1, Nz)), np.zeros((Nx, Ny, Nz+1))

#diagnostics
def kinetic_energy(u, v, w):
    uc = avg_u_to_center(u); vc = avg_v_to_center(v); wc = avg_w_to_center(w)
    return 0.5 * np.mean(uc**2 + vc**2 + wc**2)

def dissipation(u, v, w, dx, dy, dz, Re):
    uc = avg_u_to_center(u); vc = avg_v_to_center(v); wc = avg_w_to_center(w)
    dudx = (np.roll(uc, -1, axis=0) - np.roll(uc, 1, axis=0)) / (2*dx)
    dudy = (np.roll(uc, -1, axis=1) - np.roll(uc, 1, axis=1)) / (2*dy)
    dudz = (np.roll(uc, -1, axis=2) - np.roll(uc, 1, axis=2)) / (2*dz)
    dvdx = (np.roll(vc, -1, axis=0) - np.roll(vc, 1, axis=0)) / (2*dx)
    dvdy = (np.roll(vc, -1, axis=1) - np.roll(vc, 1, axis=1)) / (2*dy)
    dvdz = (np.roll(vc, -1, axis=2) - np.roll(vc, 1, axis=2)) / (2*dz)
    dwdx = (np.roll(wc, -1, axis=0) - np.roll(wc, 1, axis=0)) / (2*dx)
    dwdy = (np.roll(wc, -1, axis=1) - np.roll(wc, 1, axis=1)) / (2*dy)
    dwdz = (np.roll(wc, -1, axis=2) - np.roll(wc, 1, axis=2)) / (2*dz)
    sq = (dudx**2 + dudy**2 + dudz**2 + dvdx**2 + dvdy**2 + dvdz**2 + dwdx**2 + dwdy**2 + dwdz**2)
    return np.mean(sq) / Re

def divergence_norms(u, v, w, dx, dy, dz):
    div = divergence_mac(u, v, w, dx, dy, dz)
    return np.max(np.abs(div)), np.linalg.norm(div.ravel()) / math.sqrt(div.size)

def vorticity_fields(u, v, w, dx, dy, dz):
    uc = avg_u_to_center(u); vc = avg_v_to_center(v); wc = avg_w_to_center(w)
    dwy_dy = (np.roll(wc, -1, axis=1) - np.roll(wc, 1, axis=1)) / (2*dy)
    dv_dz = (np.roll(vc, -1, axis=2) - np.roll(vc, 1, axis=2)) / (2*dz)
    wx = dwy_dy - dv_dz
    duz_dz = (np.roll(uc, -1, axis=2) - np.roll(uc, 1, axis=2)) / (2*dz)
    dw_dx = (np.roll(wc, -1, axis=0) - np.roll(wc, 1, axis=0)) / (2*dx)
    wy = duz_dz - dw_dx
    dv_dx = (np.roll(vc, -1, axis=0) - np.roll(vc, 1, axis=0)) / (2*dx)
    du_dy = (np.roll(uc, -1, axis=1) - np.roll(uc, 1, axis=1)) / (2*dy)
    wz = dv_dx - du_dy
    mag = np.sqrt(wx*wx + wy*wy + wz*wz)
    return mag, (wx, wy, wz)

def energy_spectrum(u, v, w, grid):
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    uc = avg_u_to_center(u); vc = avg_v_to_center(v); wc = avg_w_to_center(w)
    Uhat = fftn(uc); Vhat = fftn(vc); What = fftn(wc)
    Ek = 0.5 * (np.abs(Uhat)**2 + np.abs(Vhat)**2 + np.abs(What)**2)
    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    kz = np.fft.fftfreq(Nz) * Nz
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()
    Ek_r = Ek.ravel()
    kmax = int(np.ceil(Kmag.max()))
    E_shell = np.zeros(kmax+1)
    counts = np.zeros(kmax+1, dtype=int)
    km = np.rint(Kmag).astype(int)
    for i, kbin in enumerate(km):
        if kbin <= kmax:
            E_shell[kbin] += Ek_r[i]
            counts[kbin] += 1
    return np.arange(len(E_shell)), E_shell, counts
#initial conditions
def ic_taylor_green(Nx, Ny, Nz, Lx, Ly, Lz, k0=1, amplitude=1.0):
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    z = np.linspace(0, Lz, Nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    u_c = amplitude * np.sin(k0*X) * np.cos(k0*Y) * np.cos(k0*Z)
    v_c = -amplitude * np.cos(k0*X) * np.sin(k0*Y) * np.cos(k0*Z)
    w_c = np.zeros_like(u_c)
    u = np.zeros((Nx+1, Ny, Nz)); u[1:-1,:,:] = 0.5*(u_c[0:-1,:,:] + u_c[1:,:,:]); u[0,:,:]=0.5*(u_c[-1,:,:]+u_c[0,:,:]); u[-1,:,:]=u[0,:,:]
    v = np.zeros((Nx, Ny+1, Nz)); v[:,1:-1,:] = 0.5*(v_c[:,0:-1,:] + v_c[:,1:,:]); v[:,0,:]=0.5*(v_c[:,-1,:]+v_c[:,0,:]); v[:,-1,:]=v[:,0,:]
    w = np.zeros((Nx, Ny, Nz+1))
    p = np.zeros((Nx, Ny, Nz))
    return u, v, w, p

def ic_abc(Nx, Ny, Nz, Lx, Ly, Lz, A=1.0, B=1.0, C=1.0):
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    z = np.linspace(0, Lz, Nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    u_c = A * np.sin(Z) + C * np.cos(Y)
    v_c = B * np.sin(X) + A * np.cos(Z)
    w_c = C * np.sin(Y) + B * np.cos(X)
    u = np.zeros((Nx+1, Ny, Nz)); u[1:-1,:,:] = 0.5*(u_c[0:-1,:,:] + u_c[1:,:,:]); u[0,:,:]=0.5*(u_c[-1,:,:]+u_c[0,:,:]); u[-1,:,:]=u[0,:,:]
    v = np.zeros((Nx, Ny+1, Nz)); v[:,1:-1,:] = 0.5*(v_c[:,0:-1,:] + v_c[:,1:,:]); v[:,0,:]=0.5*(v_c[:,-1,:]+v_c[:,0,:]); v[:,-1,:]=v[:,0,:]
    w = np.zeros((Nx, Ny, Nz+1)); w[:,:,1:-1] = 0.5*(w_c[:,:,0:-1] + w_c[:,:,1:]); w[:,:,0]=0.5*(w_c[:,:,-1]+w_c[:,:,0]); w[:,:,-1]=w[:,:,0]
    p = np.zeros((Nx, Ny, Nz))
    return u, v, w, p

def ic_kolmogorov(Nx, Ny, Nz, Lx, Ly, Lz, k=1, amplitude=1.0):
    u = np.zeros((Nx+1, Ny, Nz)); v = np.zeros((Nx, Ny+1, Nz)); w = np.zeros((Nx, Ny, Nz+1)); p = np.zeros((Nx, Ny, Nz))
    return u, v, w, p

def ic_isotropic_random(Nx, Ny, Nz, Lx, Ly, Lz, k0=4, amplitude=1.0, seed=0):
    rng = np.random.RandomState(seed)
    shape = (Nx, Ny, Nz)
    Uhat = (rng.normal(size=shape) + 1j*rng.normal(size=shape))
    Vhat = (rng.normal(size=shape) + 1j*rng.normal(size=shape))
    What = (rng.normal(size=shape) + 1j*rng.normal(size=shape))
    kx = 2*np.pi * fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi * fftfreq(Ny, d=Ly/Ny)
    kz = 2*np.pi * fftfreq(Nz, d=Lz/Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[K2==0] = 1.0
    kdotU = KX*Uhat + KY*Vhat + KZ*What
    Uhat = Uhat - (kdotU * KX) / K2
    Vhat = Vhat - (kdotU * KY) / K2
    What = What - (kdotU * KZ) / K2
    uc = np.real(ifftn(Uhat)); vc = np.real(ifftn(Vhat)); wc = np.real(ifftn(What))
    maxvel = max(np.max(np.abs(uc)), np.max(np.abs(vc)), np.max(np.abs(wc)), 1e-16)
    scale = amplitude / maxvel
    uc *= scale; vc *= scale; wc *= scale
    u = np.zeros((Nx+1, Ny, Nz)); u[1:-1,:,:] = 0.5*(uc[0:-1,:,:] + uc[1:,:,:]); u[0,:,:] = 0.5*(uc[-1,:,:] + uc[0,:,:]); u[-1,:,:] = u[0,:,:]
    v = np.zeros((Nx, Ny+1, Nz)); v[:,1:-1,:] = 0.5*(vc[:,0:-1,:] + vc[:,1:,:]); v[:,0,:] = 0.5*(vc[:,-1,:] + vc[:,0,:]); v[:,-1,:] = v[:,0,:]
    w = np.zeros((Nx, Ny, Nz+1)); w[:,:,1:-1] = 0.5*(wc[:,:,0:-1] + wc[:,:,1:]); w[:,:,0] = 0.5*(wc[:,:,-1] + wc[:,:,0]); w[:,:,-1] = w[:,:,0]
    p = np.zeros((Nx, Ny, Nz))
    return u, v, w, p

def ic_mms_example(Nx, Ny, Nz, Lx, Ly, Lz, t0=0.0):
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    z = np.linspace(0, Lz, Nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    uc = np.sin(X + t0); vc = -np.sin(X + t0); wc = np.zeros_like(uc)
    u = np.zeros((Nx+1, Ny, Nz)); u[1:-1,:,:] = 0.5*(uc[0:-1,:,:] + uc[1:,:,:]); u[0,:,:]=0.5*(uc[-1,:,:]+uc[0,:,:]); u[-1,:,:]=u[0,:,:]
    v = np.zeros((Nx, Ny+1, Nz)); v[:,1:-1,:] = 0.5*(vc[:,0:-1,:] + vc[:,1:,:]); v[:,0,:] = 0.5*(vc[:,-1,:] + vc[:,0,:]); v[:,-1,:] = v[:,0,:]
    w = np.zeros((Nx, Ny, Nz+1)); p = np.zeros((Nx, Ny, Nz))
    return u, v, w, p

IC_REG = {
    'taylor-green': ic_taylor_green,
    'ABC': ic_abc,
    'kolmogorov': ic_kolmogorov,
    'isotropic': ic_isotropic_random,
    'mms': ic_mms_example
}

#blowup detectors
def fit_powerlaw_blowup(time_arr, omega_arr):
    res = {'T_star': None, 'alpha': None, 'r2': None, 'valid': False}
    if len(time_arr) < 4:
        return res
    t = np.array(time_arr); w = np.array(omega_arr)
    if np.any(w <= 0):
        return res
    dt = np.diff(t); dw = np.diff(w)
    omega_dot = dw[-1] / (dt[-1] + 1e-20)
    if omega_dot <= 0:
        return res
    T_est = t[-1] - w[-1] / (omega_dot + 1e-20)
    if T_est <= t[-1] or not np.isfinite(T_est):
        return res
    tau = T_est - t
    mask = tau > 0
    if np.sum(mask) < 3:
        return res
    x = np.log(tau[mask]); y = np.log(w[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, c = np.linalg.lstsq(A, y, rcond=None)[0]
    alpha = -slope
    y_pred = slope * x + c
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-20
    r2 = 1 - ss_res/ss_tot
    res.update({'T_star': float(T_est), 'alpha': float(alpha), 'r2': float(r2), 'valid': True})
    return res

def compute_spectral_pileup(u, v, w, grid):
    k_bins, E_shell, counts = energy_spectrum(u, v, w, grid)
    total = np.sum(E_shell)
    if total <= 0 or len(E_shell) <= 1:
        return 0.0, (k_bins, E_shell, counts)
    ratio = E_shell[-1] / total
    return float(ratio), (k_bins, E_shell, counts)

def save_crash_report(prefix, step, t, u, v, w, p, diagnostics_history, fit_result, grid, cfg):
    outdir = cfg.get('blowup_crash_dump_dir', 'crash_dumps')
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{prefix}_crash_step{step:06d}_{timestamp}.npz"
    fullpath = os.path.join(outdir, fname)
    history = diagnostics_history.copy()
    vor_mag, _ = vorticity_fields(u, v, w, grid['dx'], grid['dy'], grid['dz'])
    idx = np.unravel_index(np.argmax(vor_mag), vor_mag.shape)
    r = cfg.get('blowup_local_patch_radius', 8)
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    i0, j0, k0 = idx
    i1, i2 = max(0, i0-r), min(Nx, i0+r+1)
    j1, j2 = max(0, j0-r), min(Ny, j0+r+1)
    k1, k2 = max(0, k0-r), min(Nz, k0+r+1)
    uc = avg_u_to_center(u); vc = avg_v_to_center(v); wc = avg_w_to_center(w)
    patch = {
        'u_patch': uc[i1:i2, j1:j2, k1:k2],
        'v_patch': vc[i1:i2, j1:j2, k1:k2],
        'w_patch': wc[i1:i2, j1:j2, k1:k2],
        'vor_patch': vor_mag[i1:i2, j1:j2, k1:k2],
        'patch_indices': (i1,i2,j1,j2,k1,k2),
        'vor_global_max_index': idx
    }
    np.savez_compressed(fullpath,
                        u=u, v=v, w=w, p=p,
                        diagnostics_history=history,
                        fit_result=fit_result,
                        grid=grid,
                        config=cfg,
                        local_patch=patch)
    return fullpath

def detect_blowup_nondivergence(diagnostics_history, grid, u, v, w, cfg):
    reasons = []
    info = {}
    recent = cfg.get('blowup_check_window', 10)
    times = diagnostics_history['time'][-recent:]
    omegas = diagnostics_history['omega_max'][-recent:]
    bkms = diagnostics_history['bkm'][-recent:]
    energies = diagnostics_history['energy'][-recent:]
    if len(times) < 4:
        return False, reasons, info
    # BKM
    if bkms[-1] >= cfg.get('bkm_alert_threshold', 1e-1):
        reasons.append(f"BKM exceeded threshold: {bkms[-1]:.3e} >= {cfg.get('bkm_alert_threshold')}")
    if omegas[0] > 0:
        growth = omegas[-1] / (omegas[0] + 1e-20)
        if growth >= cfg.get('omega_growth_factor', 5.0):
            reasons.append(f"omega grew by factor {growth:.2f} over last {len(omegas)} steps")
    dt_window = times[-1] - times[0]
    omega_dot = (omegas[-1] - omegas[0]) / (dt_window + 1e-20)
    if omega_dot >= cfg.get('omega_min_growth_rate', 1e-3):
        reasons.append(f"omega growth rate {omega_dot:.3e} >= {cfg.get('omega_min_growth_rate')}")
    pileup_ratio, spectrum = compute_spectral_pileup(u, v, w, grid)
    if pileup_ratio >= cfg.get('spectral_pileup_ratio', 1e-3):
        reasons.append(f"spectral pileup detected: E(k_max)/E_total={pileup_ratio:.3e}")
    fit = fit_powerlaw_blowup(times, omegas)
    if fit['valid']:
        info['fit'] = fit
        if fit['r2'] > 0.8 and fit['alpha'] > 0.5:
            reasons.append(f"power-law fit suggests blowup: alpha={fit['alpha']:.3f}, T*={fit['T_star']:.5e}, R2={fit['r2']:.3f}")
    flag = len(reasons) > 0
    info.update({'pileup_ratio': pileup_ratio, 'omega_dot': omega_dot})
    return flag, reasons, info

#I/O helpers (or helper)
def save_state_npz(filename, u, v, w, p, time_val, step, diagnostics, config):
    np.savez_compressed(filename,
                        u=u, v=v, w=w, p=p,
                        time=time_val, step=step,
                        diagnostics=diagnostics,
                        config=config)
    print(f"[IO] saved {filename}")


#time stepper & main

#batch sweeper

#parser

#entrypoint