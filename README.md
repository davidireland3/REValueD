# REValueD

This is the repo for the paper "[REValueD: Regularised Ensemble Value-Decomposition for Factorisable Markov Decision Processes](https://arxiv.org/abs/2401.08850)", to appear at ICLR 2024. The paper can be found [here](https://arxiv.org/abs/2205.10106).

`algorithms.py` contains DecQN as a baseline with REValueD incorporating the ensemble -- this repo contains on single threaded variant for easier accessibility for experimentation. 

`environment_utils.py` includes a gym wrapper for the DM Control Suite with wrappers that discretise the continuous action space. 


TODO: 

- Add regulariser to REValueD.
- Add prioritised replay buffer.
- If sufficient interest, release version using parallel workers in Ray. 
