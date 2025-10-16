# Kuramoto CPG for Quadruped Trot (Sensorless)

**What this is**  
A masters-level, bio-inspired locomotion controller implemented purely in code (no sensors). It models a **Central Pattern Generator (CPG)** using the **Kuramoto oscillator** framework to coordinate a **trot gait** for a quadruped. The controller produces synchronized phases and open-loop joint trajectories (hips/knees) and visualizes
- phase locking,
- footfall (stance) diagram,
- joint angle waveforms.

**Why it is MSc-level**  
- Nonlinear dynamics & synchronization (Kuramoto model).  
- Bio-robotics link to spinal CPGs and interlimb coordination.  
- Formal encoding of desired inter-oscillator **phase offsets** to realize a gait pattern.  
- Quantitative evaluation via phase differences.

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
