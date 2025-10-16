#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor-augmented Kuramoto CPG for quadruped gaits (trot by default)

Run:
    pip install numpy pandas matplotlib
    python cpg_with_sensors_from_dataset.py --csv your_dataset.csv
If --csv is omitted, a synthetic dataset is generated and used.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

NAMES = ["LF","RF","LH","RH"]
LF, RF, LH, RH = 0,1,2,3

def detect_deg_to_rad(df, cols):
    for c in cols:
        if c in df.columns:
            if df[c].abs().max() > 2*np.pi*1.2:  # simple heuristic
                df[c] = np.deg2rad(df[c])
    return df

def rising_edges(x):
    x = np.asarray(x).astype(int)
    return np.where((x[1:] == 1) & (x[:-1] == 0))[0] + 1

def load_or_synthesize(csv_path=None, T=20.0, freq=1.0, dt=0.005, seed=123):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'time' in df.columns:
            df['time'] = df['time'] - df['time'].iloc[0]
        else:
            df['time'] = np.arange(len(df))*dt
        angle_cols = [f'hip_{n}' for n in NAMES] + [f'knee_{n}' for n in NAMES]
        df = detect_deg_to_rad(df, angle_cols)
        return df

    # synthetic dataset (trot)
    steps = int(T/dt)
    t = np.arange(steps)*dt
    phase = 2*np.pi*freq*t
    contact_LF = (np.sin(phase) > 0).astype(int)
    contact_RH = (np.sin(phase) > 0).astype(int)
    contact_RF = (np.sin(phase + np.pi) > 0).astype(int)
    contact_LH = (np.sin(phase + np.pi) > 0).astype(int)
    hip_amp = np.deg2rad(20); knee_amp = np.deg2rad(30); psi = -np.pi/3
    rng = np.random.default_rng(seed)
    hip_LF = hip_amp*np.sin(phase) + 0.05*rng.normal(size=steps)
    hip_RF = hip_amp*np.sin(phase+np.pi) + 0.05*rng.normal(size=steps)
    hip_LH = hip_amp*np.sin(phase+np.pi) + 0.05*rng.normal(size=steps)
    hip_RH = hip_amp*np.sin(phase) + 0.05*rng.normal(size=steps)
    knee_LF = knee_amp*np.sin(phase+psi) + 0.05*rng.normal(size=steps)
    knee_RF = knee_amp*np.sin(phase+np.pi+psi) + 0.05*rng.normal(size=steps)
    knee_LH = knee_amp*np.sin(phase+np.pi+psi) + 0.05*rng.normal(size=steps)
    knee_RH = knee_amp*np.sin(phase+psi) + 0.05*rng.normal(size=steps)
    imu_ax = 0.5*np.sin(phase) + 0.1*rng.standard_normal(steps)
    imu_ay = 0.1*rng.standard_normal(steps)
    imu_az = 9.81 + 0.2*rng.standard_normal(steps)
    imu_gx = 0.4*np.cos(phase) + 0.1*rng.standard_normal(steps)

    df = pd.DataFrame({
        'time': t,
        'contact_LF': contact_LF, 'contact_RF': contact_RF,
        'contact_LH': contact_LH, 'contact_RH': contact_RH,
        'hip_LF': hip_LF, 'hip_RF': hip_RF, 'hip_LH': hip_LH, 'hip_RH': hip_RH,
        'knee_LF': knee_LF, 'knee_RF': knee_RF, 'knee_LH': knee_LH, 'knee_RH': knee_RH,
        'imu_ax': imu_ax, 'imu_ay': imu_ay, 'imu_az': imu_az, 'imu_gx': imu_gx
    })
    df.to_csv("synthetic_gait_dataset.csv", index=False)
    return df

def simulate_cpg_with_feedback(df, freq0=1.0, K=6.0):
    t = df['time'].values
    dt_series = np.diff(t, prepend=t[0])
    n = 4
    omega = 2*np.pi*freq0 * np.ones(n)
    phi = np.zeros((n,n))
    pairs_zero = [(LF,RH),(RH,LF),(RF,LH),(LH,RF)]
    pairs_pi   = [(LF,RF),(RF,LF),(LH,RH),(RH,LH),(LF,LH),(LH,LF),(RF,RH),(RH,RF)]
    for i,j in pairs_zero: phi[i,j] = 0.0
    for i,j in pairs_pi:   phi[i,j] = np.pi

    rng = np.random.default_rng(7)
    theta = rng.uniform(0, 2*np.pi, size=n)
    theta_hist = np.zeros((len(t), n))

    # Feedback gains
    phase_stance_onset = 0.0
    alpha_freq = 0.5   # frequency adaptation
    beta_phase = 0.1   # hip-based phase-slip correction

    contact_cols = [f'contact_{n}' for n in NAMES]
    has_contact = [c in df.columns for c in contact_cols]
    contact_edges = {}
    for name, present in zip(NAMES, has_contact):
        if present: contact_edges[name] = set(rising_edges(df[f'contact_{name}'].values).tolist())
        else:       contact_edges[name] = set()

    last_contact_time = {name: None for name in NAMES}
    hip_cols = [f'hip_{n}' for n in NAMES]
    has_hip = [c in df.columns for c in hip_cols]

    for k in range(len(t)):
        dt = dt_series[k] if dt_series[k] > 0 else (t[1]-t[0] if len(t)>1 else 0.01)
        # Kuramoto
        dtheta = omega.copy()
        for i in range(n):
            coupling_sum = 0.0
            for j in range(n):
                if i == j: continue
                coupling_sum += np.sin(theta[j] - theta[i] - phi[i,j])
            dtheta[i] += (K/n) * coupling_sum
        theta = (theta + dtheta*dt) % (2*np.pi)

        # (A) Phase reset on contact
        for i, name in enumerate(NAMES):
            if k in contact_edges[name]:
                theta[i] = phase_stance_onset
                if last_contact_time[name] is not None:
                    obs_period = t[k] - last_contact_time[name]
                    if obs_period > 1e-3:
                        omega[i] += alpha_freq * (2*np.pi/obs_period - omega[i])
                last_contact_time[name] = t[k]

        # (C) Phase-slip correction from hip
        if all(has_hip) and k>0:
            for i, name in enumerate(NAMES):
                h_now = df[f'hip_{name}'].iloc[k]
                h_prev = df[f'hip_{name}'].iloc[k-1]
                dh = (h_now - h_prev) / dt
                est_phase = np.arctan2(dh, h_now + 1e-6)
                diff = np.arctan2(np.sin(est_phase - theta[i]), np.cos(est_phase - theta[i]))
                theta[i] = (theta[i] + beta_phase*diff) % (2*np.pi)

        theta_hist[k,:] = theta

    return theta_hist

def summarize_and_plot(df, theta_hist):
    t = df['time'].values
    n = theta_hist.shape[1]
    plt.figure(figsize=(10,4))
    for i in range(n):
        plt.plot(t, np.unwrap(theta_hist[:,i]), label=NAMES[i])
    plt.xlabel("Time (s)"); plt.ylabel("Unwrapped phase (rad)")
    plt.title("Sensor-augmented Kuramoto CPG (dataset feedback)")
    plt.legend(); plt.tight_layout(); plt.show()

    has_contact_any = any([f'contact_{nm}' in df.columns for nm in NAMES])
    if has_contact_any:
        plt.figure(figsize=(10,3))
        for i, name in enumerate(NAMES):
            if f'contact_{name}' in df.columns:
                y = np.full(len(t), n-1-i)
                in_stance = False; start=0
                contact = df[f'contact_{name}'].values.astype(int)
                for k in range(len(t)):
                    if contact[k]==1 and not in_stance:
                        in_stance=True; start=k
                    if (contact[k]==0 and in_stance) or (k==len(t)-1 and in_stance):
                        end = k if contact[k]==0 else k
                        plt.plot(t[start:end], y[start:end], linewidth=5)
                        in_stance=False
        plt.yticks(range(n), NAMES[::-1])
        plt.xlabel("Time (s)"); plt.title("Footfall (from dataset contacts)")
        plt.tight_layout(); plt.show()

    # Stats
    dt = t[1]-t[0] if len(t)>1 else 0.01
    transient = int(3.0/max(dt,1e-3))
    def mean_phase_diff(i,j):
        return np.angle(np.exp(1j*(theta_hist[transient:,i]-theta_hist[transient:,j]))).mean()
    stats = {
        "LF-RH": mean_phase_diff(LF,RH),
        "RF-LH": mean_phase_diff(RF,LH),
        "LF-RF": mean_phase_diff(LF,RF),
        "LH-RH": mean_phase_diff(LH,RH)
    }
    print("Mean phase differences (rad) after feedback:", stats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="", help="Path to gait dataset CSV (optional)")
    args = parser.parse_args()

    df = load_or_synthesize(args.csv if args.csv else None)
    theta_hist = simulate_cpg_with_feedback(df)
    summarize_and_plot(df, theta_hist)

if __name__ == "__main__":
    main()
