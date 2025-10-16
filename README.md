# Kuramoto CPG for Quadruped Trot (Sensorless)

**What this is**  
A bio-inspired locomotion controller implemented purely in code (no sensors). It models a **Central Pattern Generator (CPG)** using the **Kuramoto oscillator** framework to coordinate a **trot gait** for a quadruped. The controller produces synchronized phases and open-loop joint trajectories (hips/knees) and visualizes
- phase locking,
- footfall (stance) diagram,
- joint angle waveforms.


**How to run**
```bash
pip install numpy matplotlib
python cpg_quadruped_trot.py
```

**Extensions (optional)**
- Change offsets to pace/walk/gallop and compare stability.  
- Add noise to frequencies and study robustness.  
- Replace sin coupling with Hopf oscillators + amplitude dynamics.  
- Map joint signals to a simple physics sim (e.g., PyBullet) for open-loop playback.
