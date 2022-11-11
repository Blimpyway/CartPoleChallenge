Here-s an attempt to win OpenAI's gym CartPole contest combining Numenta's Sparse Distributed Representations with bitpair value maps. 

The two versions - cp_vmaps_fast.py and cp_vmap_ovenc.py - differ only through the encoding method used to transform the 4 value observation space into a SDR representation which is further used by populating a ValueMap with Q-values. 
1. cp_vmaps_fast.py uses a simple scalar encoder where each observation value gets its own 1/4 allotment of the available SDR space. This one is both computationally inexpensive and more robust/reliable.  Episodes run to solve the CartPole-v1 are ~124 with ~26 failed (ended before 500 steps). Out of 1000 training sessions all converge in less than 300 episodes,  
2. cp_vmap_ovenc.py uses an overlapping encoding which makes it both slower to compute and more brittle - 1-2% of trials do not converge in 1000 episodes.  However there are also a significant % of training sessions that solve the cartpole in 100 episodes (minimum possible)  and some of them with  only 2 failed episodes. 
Details of this encoding are discussed in Numenta's Forum https://discourse.numenta.org/t/scalar-vectors-as-intermediate-stages/10259/2?u=cezar_t . The encoder object (def CycleEncoder) source is relatively simple too. 

Code is a bit crowded since I avoided external dependencies - no ML imports, only numpy and gym are needed. 

$ pip install numpy gym 

For important execution speed improvement you should also use numba: 

$ pip install numba

-----------------

Short guide for cp_vmaps_fast.py - the other one is almost the same, only the SDR encoder part being different: 

- Lines 44-92 
  A simple SDR scalar encoder that converts the 4 environment observations to a smallish 16/68 bit SDR
  The only added "feature" is, contrary to Numenta's recomnendations, the ability to dynamically scale so it fits exactly to observed environment state. The dynamic feature on delays the solving with a few (5?) episodes. 
  The SDR can be shrinked to 12/40bit with a slight increase in the number of episodes till solving (+5?), at this size code should fit - and train in real time in a Arduino Uno. Not just "inference"   
  
- Lines 94-128 
  A simplified bitpair ValueMap incarnation which simply maps a SDR to a single scalar Q-value, with negative values telling the agent to push the cartpole left, and positive ones to move right 
  
- Following lines up to 238
  The agent class itself with methods to train() and show() results. Unlike most RL solvers, it only uses the past 10 observation in failed runs. So it ignores any succesfull run of 500 step()-s , or any step preceding the last 10 before falling.

- Lines 242-252 attempt to numba-njit compile most demanding functions

- Lines 255-265 featuring how the agent can be used. 

Enjoy!
Cezar Totth
