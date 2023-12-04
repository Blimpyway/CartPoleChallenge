"""
Cartpole using value maps example. 

The only dependencies are gym, numpy and optionally numba.
If by any chance numba is installed, avoid blinking ;-)

Average episodes needed to solve the environment is under 125.  

By OpenAI's rules, the environment is considered solved when average no.of steps over the last 
100 episodes exceeds 195 in CartPole-v0 or 475 in v1

Average no. of failed episodes before solving is under 26 in v1.

--------------
For HTM folks 

Encoder is a simple SDR scalar encoder which transforms the 4 value observations (aka env state) in
a 16/68 SDR (68 is total SDR size, 16 bits are on) 
Each state scalar is assigned an 1/4 slice of 4/17 bits which counts as a 14 value bins (VALUE_STEPS parameter)
The encoder maximum values expand dynamically initial all 4 scalars are capped at 0.2

The values of "danger" are encoded in a simple bitpair value map over the 68 bit sized SDR. 
Negative/Positive values means danger to move left/right in the state represented by the SDR.  

-------------
Copyright: blimpyway aka cezar_t

"""

import gym, random
import numpy as np
VALUE_STEPS = 14      # number of steps for state values
VALUE_BITS  =  4      # SDR ON bits for each state value 

# VALUE_STEPS, VALUE_BITS = 8,2
DANGER_STEPS = 12     # These many steps preceding failure will be mapped as danger teritory
DANGER_START = 10

CART_POLE = "CartPole-v1"       # solved in under 126 episodes over 1000 tests
# CART_POLE = "CartPole-v0"     # solved in ~115 episodes over 1000 tests

class Self: pass # Pseudo classes are cute

########################## Encoder stuff, converts observation values to SDR  
########################## there is a numba and non-numba variant

# The fast_encode is njit-able by numba
def fast_encode(state, max_vals, state_steps, state_bits, state_width):
    for greater in np.where(np.abs(state) > max_vals):
        max_vals[greater] = np.abs(state)[greater]
    step_size = max_vals * 2 / state_steps
    int_state = ((state + max_vals) / step_size - 0.5).astype(np.int32)
    r = []
    for i,v in enumerate(int_state): 
        for bit in range(state_bits):
            r.append(state_width*i+v+bit)
    return np.uint32(r)

# This will be used (njit-compiled) if numba can be imported
def _state2sdr(state_steps, state_bits, num_states):
    state_width = state_steps + state_bits - 1
    sdr_size = state_width * num_states
    # max_vals = np.ones(4) / 5
    max_vals = np.array([4.796713  , 6.4722023 , 0.41887885, 6.852011])/2
    # max_vals = np.ones(4) - np.inf
    encode = lambda state: fast_encode(state, max_vals, state_steps, state_bits, state_width)
    return sdr_size, encode

# An auto-calibrating observation to sdr converter. 
# Observed values are assumed to be "symmetrical" around 0.0 
def state2sdr(state_steps, state_bits, num_states): 
    state_width = state_steps + state_bits - 1 
    sdr_size = state_width * num_states 
    max_vals = np.ones(4) / 5
    # max_vals = np.ones(4) - np.inf # if one complains the above initialisation is handcrafted...
    starts = np.arange(num_states) * state_width
    out = np.zeros(num_states*state_bits,dtype = np.int32)
    for s in range(num_states): 
        for b in range(state_bits): 
            out[s*state_bits+b] = state_width * s + b
    def encode(state): 
        bigger = max_vals < np.abs(state)
        max_vals[bigger] = np.abs(state[bigger])
        step_size = max_vals * 2 / state_steps
        int_state = ((state + max_vals) / step_size - 0.5).astype(np.int32)
        oout = out.copy()
        for i in range(num_states): 
            ostart = i * state_bits
            oout[ostart:ostart+state_bits] += int_state[i]
        return oout   
    return sdr_size, encode



######################################################################################3
###### Simplified ValueMap code

def address_list(sdr): 
    """
    Converts a SDR to a list of value map addresses
    """
    ret = []
    for i in range(1, len(sdr)):
        ival = (sdr[i]*(sdr[i]-1))//2
        for j in range(i):
            ret.append((ival + sdr[j]))
    return ret

def add_val(mem, sdr, value): 
    for a in address_list(sdr):
        mem[a] += value


def read_val(mem, sdr): 
    total = 0
    for a in address_list(sdr):
        total += mem[a]
    return total

######## Actual ValueMap (pseudo)class. The above address_list, add_val and read_val functions
######## are kept out of so they can be njit-compiled if numba can be imported
def SimpleValueMap(sdr_size):
    my = Self() 
    my.mem = np.zeros((sdr_size * (sdr_size-1)) // 2, dtype = np.int32)
    my.add  = lambda sdr, value: add_val (my.mem, sdr, value)
    my.read = lambda sdr:        read_val(my.mem, sdr)
    return my

#########################################################################################
####### Here-s the  player pseudo class
def ValueMapPlayer(yenv, value_steps, value_bits): 
    """
    Returns a player that learns to balance the cartpole,
    parameters:
        env: A CartPole environment
        value_steps: how many steps are used to encode a state value to SDR
        value_bits:  how many ON bits are used for each state value

    """
    my = Self()
    my.env = yenv
    NUM_STATES = my.env.observation_space.sample().size

    sdr_size, to_sdr = state2sdr(value_steps, value_bits, NUM_STATES) 
    print(f"SDR size is {sdr_size}")
    steps = []  # Records (state-SDR, action) for every step in an episode
    my.episodes = [] # Records number of steps for each learning episodes
    my.vmap = SimpleValueMap(sdr_size) # Maps the dangerous teritory

    def pick_action(obs):
        """
        Chosing an action from observation
        """
        # print(f"Observation in pick_action() is {obs}, type(): {type(obs)}")
        # print(">", flush = True, end = '')
        sdr = to_sdr(obs)
        danger = my.vmap.read(sdr)
        if danger == 0: 
            action = random.randint(0,1)
        else:
            # aka if danger is "-" action is 1, if danger is "+" action is 0
            action = int(danger < 0)
        steps.append((sdr, action)) # Record (sdr state, action) for learning
        return action
        

    def round_done(learning): 
        """
        This is called after every round. 
        learning - bool - specifies whether to enable learning for this round or not. 
            learning happens only if the round failed, which means 
            it  finished before reaching env.spec.max_episode_steps
        returns is the player has solved the env challenge, defined as an past 100 episodes
        average score greater than env.spec.reward_threshold

        """
        my.episodes.append(len(steps)) # Record the number of episode steps
        message = f"Ep. {len(my.episodes):3d} ran {my.episodes[-1]:3d} steps     "
        if len(steps) < my.env.spec.max_episode_steps: # failed, something to learn
            print("\n", message, end = " ", flush=True)
            danger = min(len(steps), DANGER_STEPS) if learning else 0  # if learn disabled, don't update vmap
            while danger: 
                sdr, action = steps.pop()
                left_or_right = int((action * 2) - 1) # Turns action into a [-1, 1] choice
                my.vmap.add(sdr, left_or_right * (DANGER_START + danger)) # ... I should explain..
                # my.vmap.add(sdr, left_or_right * danger) 
                danger -= 1
        else:
            print("+", end = '') # A '+' means episode is a success!
            # print(message, end = '\r')

        steps.clear()
        solved = len(my.episodes) >= 100 and np.mean(my.episodes[-100:]) > my.env.spec.reward_threshold

        return solved

    def one_round(learning = True):
        """
        Runs a single round
        """
        done = False
        obs, whatever = my.env.reset()
        while not done:
            action = pick_action(obs)
            obs,reward,done,trunc,info = my.env.step(action)
            done = done or trunc
        return round_done(learning)

    def show(): 
        """
        Runs a rendered round without training. 
        """
        senv = my.env
        my.env = gym.make(senv.spec.id, render_mode = 'human')
        success = one_round(learning = False)
        my.env.close() 
        my.env = senv
        print()
        return success

    def train(render = False, max_episodes = 1000):
        """
        returns number of episodes needed to train
        Arguments:
            render:     whether to show env during training (this easily gets boring)
            max_episodes: give up if environment is not solved thus far.
        """
        for ep in range(max_episodes):
            solved = one_round(learning = True) 
            if solved:
                break 
        print('\n')

        total_episodes  = ep+1
        total_steps     = np.sum(my.episodes) 
        failed_episodes = (np.array(my.episodes) < my.env.spec.max_episode_steps).sum()
        if solved: 
            print(f"Solved in {total_episodes} episodes, {total_steps} steps")
            print(f" {failed_episodes} of these episodes failed hence were needed to learn")
        else:
            print(f"Failed to solve in {max_episodes} episodes, {np.sum(my.episodes)} steps")
        
        return total_episodes, failed_episodes, total_steps

    # make my a proper object
    my.one_round, my.show, my.train =  one_round,    show,    train
    return my



############# Attempting to numba compile thing
try:
    from numba import njit
    state2sdr       = _state2sdr
    fast_encode     = njit(fast_encode,     cache = True)
    address_list    = njit(address_list,    cache = True)
    add_val         = njit(add_val,         cache = True)
    read_val        = njit(read_val,        cache = True)
except Exception:
    print("Can't import numba, I will be 10-15 times slower..")



# Shortcut for interactive mode
Player = lambda: ValueMapPlayer(gym.make(CART_POLE), value_steps = VALUE_STEPS, value_bits = VALUE_BITS)

p = Player() 

episodes = p.train()

print("Let's show off..")

p.show()

"""
# Example results
{
 'trials': 1000,
 'failed episodes min, max, mean': (3, 112, 26.352), # some trials learn after 3 failed episodes
 'time msec': 643,
 'tsteps': 52297739,
 'results min, max, mean': (100, 283, 124.521)
}

# With the settings:

VALUE_STEPS = 14      # number of steps for state values
VALUE_BITS  =  4      # SDR ON bits for each state value

DANGER_STEPS = 10     # These many steps preceding failure will be mapped as danger teritory


"""
