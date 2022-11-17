"""
Cartpole using value maps and overlapping SDR encoder example. 

The only dependencies are gym, numpy and optionally numba.
If by any chance numba is installed, avoid blinking ;-)

By OpenAI's rules, the environment is considered solved when average no.of steps over the last 
100 episodes exceeds 195 in CartPole-v0 or 475 in v1

--------------
For HTM folks 

The encoder converts each observation value to a scalar vector, the for scalar vectors are summed and 
the SDR is obtained by picking smallest P values  
See CycleEncoder function below and https://discourse.numenta.org/t/scalar-vectors-as-intermediate-stages/10259?u=cezar_t


The values of "danger" are encoded in a simple bitpair value map over the 68 bit sized SDR. 
Negative/Positive values means danger to move left/right in the state represented by the SDR.  

-------------
Copyright: blimpyway aka cezar_t

"""

# import gymnasium as gym
import gym
import random
import numpy as np

# Different options for encoding SDR. Too sparse converge slow, too dense tend to be brittle
SDR_SIZE, SDR_LEN = 160, 40
SDR_SIZE, SDR_LEN =  80, 20
SDR_SIZE, SDR_LEN = 120, 30
SDR_SIZE, SDR_LEN = 120, 20
SDR_SIZE, SDR_LEN = 100, 16
SDR_SIZE, SDR_LEN =  68, 17
SDR_SIZE, SDR_LEN =  40, 10  # Still converges 98% of trials
SDR_SIZE, SDR_LEN = 160, 30

DANGER_STEPS = 14     # These many steps preceding failure will be mapped as danger teritory
DANGER_START = 10

CART_POLE = "CartPole-v1"       # solved in under 126 episodes over 1000 tests
# CART_POLE = "CartPole-v0"       # solved in ~115 episodes over 1000 tests
xenv = gym.make(CART_POLE)

class Self: pass # Pseudo classes are cute

########################## Encoder stuff, converts observation values to SDR  

def min_max_adjust(values, maxims):  # Converts values to [0,1] according to their recorded min/max window
    n = values.size
    out = np.zeros(n, dtype = np.float32)
    for i in range(n): 
        maxims[i] = max(abs(values[i]), maxims[i])
        out[i] = (values[i] + maxims[i]) / (2*maxims[i])
    return out


def CycleEncoder(sdr_size, sdr_len, num_obs): 
    v = np.arange(sdr_size) 
    vectors = []

    for i in range(num_obs): 
        np.random.shuffle(v)
        vectors.append(v.copy())

    vectors = np.array(vectors, dtype = np.uint32)

    # The choices below say whether to give the encoder a head start or not
    # maxims = np.zeros(num_obs) - np.inf    # totally dynamic no idea what obs space values would be
    # maxims = np.zeros(num_obs) + 0.2       # giving it a headstart (max angle settles a bit over 0.2
    maxims = np.array([4.796713  , 6.4722023 , 0.41887885, 6.852011]) / 2  # quite an accurate match

    def _dense(vals):
        avals = min_max_adjust(vals, maxims)
        ivals = np.uint32(avals * (sdr_size - sdr_len * 1.25))
        sums = []
        for val, vector in zip(ivals, vectors):
            sums.append((val + vector) % sdr_size)
        return np.sum(sums, axis = 0)

    def dense(vals): 
        avals = min_max_adjust(vals, maxims) 
        ivals = np.uint32(avals * (sdr_size - sdr_len * 1.25))
        sums = vectors.copy()
        for i in range(num_obs): 
            sums[i] += ivals[i]
        return np.sum(sums % sdr_size, axis = 0)

    def sdr(vals):
        d = dense(np.array(vals, dtype = np.float32))
        dvec = np.argsort(d)[:sdr_len]
        dvec.sort()
        return dvec

    return sdr

######################################################################################3
###### Simplified ValueMap code

def address_list(sdr): 
    # Converts a SDR to a list of value map addresses
    ret = []
    for i in range(1, len(sdr)):
        ival = (sdr[i]*(sdr[i]-1))//2
        for j in range(i):
            ret.append((ival + sdr[j]))
    return ret

def add_val(mem, sdr, value): 
    for a in address_list(sdr):
        mem[a] += value

def read_total(mem, sdr): 
    total = 0
    for a in address_list(sdr):
        total += mem[a]
    return total

def SimpleValueMap(sdr_size):
    '''
    Actual ValueMap (pseudo)class. The above address_list, add_val and read_total functions
    are kept out of so they can be njit-compiled if numba can be imported
    '''
    my = Self() 
    my.mem = np.zeros((sdr_size * (sdr_size-1)) // 2, dtype = np.int32)
    my.add  = lambda sdr, value: add_val (my.mem, sdr, value)
    my.read = lambda sdr:        read_total(my.mem, sdr)
    return my

#########################################################################################
####### Here-s the  player pseudo class
to_sdr = None
def ValueMapPlayer(yenv, sdr_size, sdr_len): 
    """
    Returns a player that learns to balance the cartpole,
    parameters:
        env: A CartPole environment

    """
    my = Self()
    my.env = yenv
    num_vals = my.env.observation_space.sample().size
    # global to_sdr
    # if to_sdr is None:   to_sdr = CycleEncoder(sdr_size, sdr_len, num_vals) 
    to_sdr = CycleEncoder(sdr_size, sdr_len, num_vals) 
    print(f"SDR size is {sdr_size}")
    steps = []  # Records (state-SDR, action) for every step in an episode
    my.episodes = [] # Records number of steps for each learning episodes
    my.vmap = SimpleValueMap(sdr_size) # Maps the dangerous teritory

    def pick_action(obs):
        """
        Chosing an action from observation
        """
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
                # my.vmap.add(sdr, left_or_right * danger)                  # Without a bump
                # my.vmap.add(sdr, left_or_right * 10000 * .95 ** danger)      # exponential decay. 
                danger -= 1
        else:
            print("+", end = '', flush = True) # A '+' means episode is a success!
            # print(message, end = '\r')

        steps.clear()
        solved = len(my.episodes) >= 100 and np.mean(my.episodes[-100:]) > my.env.spec.reward_threshold

        return solved

    def one_round(learning = True):
        """
        Runs a single round
        """
        done = False
        obs, stuff = my.env.reset()
        # print(f"After env.reset - obs={obs}, stuff={stuff}")
        while not done:
            action = pick_action(obs)
            # if render: my.env.render()
            obs,reward,done,truncated, info = my.env.step(action)
            done = done or truncated
            # print(f"r:{reward}, d:{done}, t:{truncated}")
        return round_done(learning)

    def show(): 
        """
        Runs a rendered round without training. 
        """
        save_env = my.env 
        my.env = gym.make(save_env.spec.id, render_mode = 'human')
        success = one_round(learning = False)
        my.env = save_env
        print()
        return success

    def train(max_episodes = 1000):
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
            print(f" {failed_episodes} of them ended early and were used to learn")
        else:
            print(f"Failed to solve in {max_episodes} episodes, {np.sum(my.episodes)} steps")
        
        return total_episodes, failed_episodes, total_steps

    # make my a proper object
    my.one_round, my.show, my.train =  one_round,    show,    train
    return my



############# Attempting to numba compile key functions
try:
    from numba import njit
    min_max_adjust  = njit(min_max_adjust,  cache = True)
    address_list    = njit(address_list,    cache = True)
    add_val         = njit(add_val,         cache = True)
    read_total      = njit(read_total,      cache = True)
except Exception:
    print("Can't import numba, I will be slow")


if __name__ == "__main__":
    import time
    # Shortcut for interactive mode
    Player = lambda: ValueMapPlayer(xenv, SDR_SIZE, SDR_LEN)

    p = Player() 

    episodes = p.train()

    print("Let's show off..", flush = True)

    p.show()

    time.sleep(2)
