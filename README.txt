Here-s an attempt to win OpenAI's gym CartPole contest combining Numenta's Sparse Distributed Representations with bitpair value maps. 
Code is a bit crowded since I avoided external dependencies - no ML imports, only numpy and gym are needed. 

$ pip install numpy gym 

For blinking speed you can use numba: 

$ pip install numba

-----------------

Short guide: 

- Lines 44-92 
  There-s a simple SDR scalar encoder that converts the 4 environment observations to a smallish 16/68 bit SDR
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
