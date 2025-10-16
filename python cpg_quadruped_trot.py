# Kuramoto-based Central Pattern Generator (CPG) for quadruped trot gait
# Author: Panagiota Grosdouli 
# Description:
#   Simulates 4 coupled phase oscillators (Kuramoto model) that synchronize to a trot gait.
#   Generates phase trajectories, footfall diagram, and open-loop hip/knee joint trajectories.
#   No sensors are used; this is a pure central pattern generator.
#
#   Biological link: CPGs in the vertebrate spinal cord coordinate rhythmic motions (e.g., walking)
#   without requiring phasic sensory input. Here we implement a simplified mathematical model of
#   such a circuit using desired phase offsets between oscillators to realize the trot.
#
# Usage:
#   python cpg_quadruped_trot.py
#
# Requirements:
#   python>=3.9, numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt

def simulate_kuramoto_cpg(T=20.0, dt=0.002, freq=1.0, K=6.0, seed=42):
    """Simulate a 4-oscillator Kuramoto CPG with trot phase-offset constraints.
    Returns time, phase history, hip and knee angle trajectories, and stance booleans.
    """
    steps = int(T/dt)
    n = 4
    names = ["LF","RF","LH","RH"]
    LF, RF, LH, RH = 0,1,2,3

    # natural frequencies ~1 Hz
    omega = 2*np.pi*freq * np.ones(n)

    # desired phase offsets phi_ij to encode trot
    phi = np.zeros((n,n))
    pairs_zero = [(LF,RH),(RH,LF),(RF,LH),(LH,RF)]  # diagonal pairs in-phase
    pairs_pi   = [(LF,RF),(RF,LF),(LH,RH),(RH,LH),(LF,LH),(LH,LF),(RF,RH),(RH,RF)]
    for i,j in pairs_zero:
        phi[i,j] = 0.0
    for i,j in pairs_pi:
        phi[i,j] = np.pi

    # init phases
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2*np.pi, size=n)

    theta_hist = np.zeros((steps,n))
    for t in range(steps):
        dtheta = omega.copy()
        for i in range(n):
            coupling_sum = 0.0
            for j in range(n):
                if i == j: 
                    continue
                coupling_sum += np.sin(theta[j] - theta[i] - phi[i,j])
            dtheta[i] += (K/n) * coupling_sum
        theta = (theta + dtheta*dt) % (2*np.pi)
        theta_hist[t,:] = theta

    # open-loop joint trajectories
    A_hip  = 20.0 * np.pi/180.0   # 20 deg
    A_knee = 30.0 * np.pi/180.0   # 30 deg
    psi    = -np.pi/3             # knee phase offset
    hip  = A_hip*np.sin(theta_hist)
    knee = A_knee*np.sin(theta_hist + psi)

    # stance heuristic: hip angle > 0 => stance (toy)
    stance = hip > 0.0

    time = np.arange(steps)*dt
    return time, theta_hist, hip, knee, stance, names

def plot_results(time, theta_hist, hip, knee, stance, names):
    n = len(names)
    # Plot 1: phase
    plt.figure(figsize=(9,4))
    for i in range(n):
        plt.plot(time, np.unwrap(theta_hist[:,i]), label=names[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Unwrapped phase (rad)")
    plt.title("Kuramoto CPG – Phase Locking to Trot (No Sensors)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # Plot 2: footfall (stance) diagram
    steps = len(time)
    plt.figure(figsize=(9,3))
    for i in range(n):
        y = np.full(steps, n-1-i)
        in_stance = False
        start = 0
        for k in range(steps):
            if stance[k,i] and not in_stance:
                in_stance = True
                start = k
            if (not stance[k,i] and in_stance) or (k == steps-1 and in_stance):
                end = k if not stance[k,i] else k
                plt.plot(time[start:end], y[start:end], linewidth=5)
                in_stance = False
    plt.yticks(range(n), names[::-1])
    plt.xlabel("Time (s)")
    plt.title("Footfall (stance) diagram – Trot gait (open-loop)")
    plt.tight_layout()
    plt.show()

    # Plot 3: hip trajectories
    plt.figure(figsize=(9,4))
    for i in range(n):
        plt.plot(time, hip[:,i], label=f"Hip {names[i]}")
    plt.xlabel("Time (s)")
    plt.ylabel("Hip angle (rad)")
    plt.title("Hip trajectories (open-loop)")
    plt.legend(loc="upper right", ncols=2)
    plt.tight_layout()
    plt.show()

    # Plot 4: knee trajectories
    plt.figure(figsize=(9,4))
    for i in range(n):
        plt.plot(time, knee[:,i], label=f"Knee {names[i]}")
    plt.xlabel("Time (s)")
    plt.ylabel("Knee angle (rad)")
    plt.title("Knee trajectories (open-loop)")
    plt.legend(loc="upper right", ncols=2)
    plt.tight_layout()
    plt.show()

def main():
    time, theta_hist, hip, knee, stance, names = simulate_kuramoto_cpg()
    # quantify phase locking (mean phase diffs after transient)
    dt = time[1]-time[0]
    transient = int(5.0/dt)
    LF, RF, LH, RH = 0,1,2,3
    mean_phase = {
        "LF-RH": np.angle(np.exp(1j*(theta_hist[transient:,0]-theta_hist[transient:,3]))).mean(),
        "RF-LH": np.angle(np.exp(1j*(theta_hist[transient:,1]-theta_hist[transient:,2]))).mean(),
        "LF-RF": np.angle(np.exp(1j*(theta_hist[transient:,0]-theta_hist[transient:,1]))).mean(),
        "LH-RH": np.angle(np.exp(1j*(theta_hist[transient:,2]-theta_hist[transient:,3]))).mean()
    }
    print("Mean phase differences (rad):", mean_phase)
    plot_results(time, theta_hist, hip, knee, stance, names)

if __name__ == "__main__":
    main()
