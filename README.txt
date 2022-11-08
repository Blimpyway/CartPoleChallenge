Here-s an attempt to win OpenAI's gym CartPole contest combining Numenta's 
Sparse Distributed Representations with bitpair value maps. 

Code is a bit crowded since I avoided external dependencies - no ML imports, only
numpy and gym are needed. 

$ pip install numpy gym 

For blinking speed you can use numba: 

$ pip install numba

-----------------
Short guide: 

- Lines 44-92 
  There-s a simple SDR scalar encoder that converts the 4 environment observations to a 
  smallish 16/68 bit SDR
  The only added "feature" is, contrary to Numenta's recomnendations, the ability to 
  dynamically scale so it fits exactly to observed environment state. 
  
- Lines 94-128 
  A simplified bitpair ValueMap incarnation which simply maps a SDR to a single 
  scalar Q-value, with negative values telling the agent to push the cartpole left, 
  and positive ones to move right 
  
- Following lines up to 238
  The agent class itself with methods to train() and show() results.

- Lines 242-252 attempt to numba-njit compile most demanding functions

- Lines 255-265 featuring how the agent can be used. 

Enjoy!
Cezar Totth
