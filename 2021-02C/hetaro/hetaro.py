"""
Komi's reinforcement learning based agent 'Hetaro'
and everything needed to train it
"""

try:
    from .ricochet import vector
    del vector
except ImportError:
    import shutil
    shutil.copyfile('ricochet.py', 'hetaro/ricochet.py')

import abc
import argparse
import numpy as np
import torch
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import pathlib
import datetime
import warnings
import random
import re
from .ricochet import vector, base_bot, hero, disc, tick_action, half_state, match_state, global_id_generator, MOVE_SPEED, DISC_SPEED, \
    DISC_COST, CHARGE_MAX, HP_INITIAL, BOUNCES_INITIAL, INTERSECT_DISTANCE, WIDTH, HEIGHT, MATCH_WIN_BONUS, MATCH_MAX_TICKS, reid_bot

class reward_tap(abc.ABC):
    """
    Abstract class for a reward tap, which manages a finite pool of reward to be given out
    for doing certain things.
    """
    @abc.abstractmethod
    def tap(self, x: float) -> float:
        pass
    @abc.abstractmethod
    def reset(self):
        pass

class lin_reward_tap(reward_tap):
    """
    Implements a reward tap that releases reward linearly.
    This means tapping 0.4 means releasing 0.4 units, or less if less than 0.4 is available.
    """
    def __init__(self, limit):
        self.limit = limit
        self.given = 0
    def tap(self, x):
        rem = self.limit - self.given
        x = max(0, min(x, rem))
        self.given += x
        return x
    def tap_upto(self, x):
        return self.tap(x - self.given)
    def reset(self):
        self.given = 0

class exp_reward_tap(reward_tap):
    """
    Implements a reward tap that releases reward exponentially.
    This means tapping 0.4 means releasing 40% of the remaining available reward.
    """
    def __init__(self, limit):
        self.limit = limit
        self.given = 0
    def tap(self, x):
        rem = self.limit - self.given
        x = max(0, x * rem)
        self.given += x
        return x
    def reset(self):
        self.given = 0

XREWARD_WEIGHT = 5

class xreward(abc.ABC):
    @abc.abstractmethod
    def reset(self):
        pass
    @abc.abstractmethod
    def evaluate(self, obs: np.ndarray, raw_action: np.ndarray):
        pass

class xreward_near(xreward):
    def __init__(self):
        self.tap = exp_reward_tap(XREWARD_WEIGHT)
    def reset(self):
        self.tap.reset()
    def evaluate(self, obs, raw_action):
        dist = np.hypot(obs[0, 0, 1] - obs[1, 0, 1], obs[0, 0, 2] - obs[1, 0, 2])
        x = max(0, 2.3 - dist) / 1000
        return self.tap.tap(x)

class xreward_far(xreward):
    def __init__(self):
        self.tap = exp_reward_tap(XREWARD_WEIGHT)
    def reset(self):
        self.tap.reset()
    def evaluate(self, obs, raw_action):
        dist = np.hypot(obs[0, 0, 1] - obs[1, 0, 1], obs[0, 0, 2] - obs[1, 0, 2])
        x = dist / 1000
        return self.tap.tap(x)

class xreward_aim(xreward):
    def __init__(self):
        self.tap = exp_reward_tap(XREWARD_WEIGHT)
    def reset(self):
        self.tap.reset()
    def evaluate(self, obs, raw_action):
        ideal_dir = vector(obs[1, 0, 1] - obs[0, 0, 1], obs[1, 0, 2] - obs[0, 0, 2])
        total = 0
        for i in range(2):
            shoot_dir = vector(raw_action[i*3+3], raw_action[i*3+4])
            dcos = ideal_dir.cosine(shoot_dir)
            x = max(0, dcos - 0.9) ** 2 / 2
            total += self.tap.tap(x)
        return total

class xreward_dodge(xreward):
    def __init__(self):
        self.tap = lin_reward_tap(1)
    def reset(self):
        self.tap.reset()
    def evaluate(self, obs, raw_action):
        x = (1 - obs[0, 0, 5]) / 2
        return -self.tap.tap_upto(x) * XREWARD_WEIGHT

class xreward_doubleshot(xreward):
    def __init__(self):
        self.tap = exp_reward_tap(XREWARD_WEIGHT)
    def reset(self):
        self.tap.reset()
    def evaluate(self, obs, raw_action):
        x = 0
        x = (1 + obs[0, 0, 6]) / 1000
        if obs[0, 0, 6] > 0.999:
            x += ((raw_action[2] > 0) + (raw_action[5] > 0)) ** 2 * 0.05
        return self.tap.tap(x)

class xreward_manydisc(xreward):
    def __init__(self):
        self.tap = exp_reward_tap(XREWARD_WEIGHT)
    def reset(self):
        self.tap.reset()
    def evaluate(self, obs, raw_action):
        x = 0
        for i in range(10):
            x += obs[1, i+1, 0] > 0
        x /= 2000
        return self.tap.tap(x)

class xreward_wastecharge(xreward):
    def __init__(self):
        self.counter = 0
        self.tap = exp_reward_tap(XREWARD_WEIGHT)
    def reset(self):
        self.counter = 0
        self.tap.reset()
    def evaluate(self, obs, raw_action):
        if obs[0, 0, 6] > 0.999:
            self.counter += 1
        else:
            self.counter = 0
        x = (self.counter >= 2) * 0.02
        return -self.tap.tap(x)

class xreward_bouncehit(xreward):
    def __init__(self):
        self.tap = exp_reward_tap(XREWARD_WEIGHT)
    def reset(self):
        self.tap.reset()
    def evaluate(self, obs, raw_action):
        total = 0
        intersect_at = INTERSECT_DISTANCE * (2 / WIDTH)
        opos = vector(obs[1, 0, 1], obs[1, 0, 2])
        for i in range(10):
            has_disc = obs[1, i+1, 0] > 0
            if has_disc:
                dpos = vector(obs[1, i+1, 1], obs[1, i+1, 2])
                would_hit = opos.distance(dpos) < intersect_at
                did_bounce = obs[1, i+1, 5] < 0.999
                if would_hit and did_bounce:
                    x = 0.2
                    total += self.tap.tap(x)
        return total

xreward_all = {
    'near': xreward_near,
    'far': xreward_far,
    'aim': xreward_aim,
    'dodge': xreward_dodge,
    'doubleshot': xreward_doubleshot,
    'manydisc': xreward_manydisc,
    'wastecharge': xreward_wastecharge,
    'bouncehit': xreward_bouncehit,
    }

def _map_range(amin, amax, bmin, bmax, a, clamp = False):
    u = (a - amin) / (amax - amin)
    if clamp:
        u = min(1, max(0, u))
    v = u * (bmax - bmin) + bmin
    return v

def _ricochet_to_agent_observation(halfs):
    result = np.zeros((2, 11, 7), dtype=np.float32)
    for i in range(2):
        ihalf = halfs[i]
        # hero / meta info
        result[i, 0, 0] = 1
        result[i, 0, 1] = _map_range(0, WIDTH, -1, 1, ihalf.hero.pos.x)
        result[i, 0, 2] = _map_range(0, WIDTH, -1, 1, ihalf.hero.pos.y)
        result[i, 0, 5] = _map_range(-1, HP_INITIAL, -1, 1, ihalf.hero.hp, True)
        result[i, 0, 6] = _map_range(0, CHARGE_MAX, -1, 1, ihalf.hero.charge, True)
        # 10 nearest discs
        idiscs = list(ihalf.discs)
        idiscs.sort(key=(lambda jdisc:ihalf.hero.pos.distance(jdisc.pos)))
        idiscs = idiscs[:10]
        for j, jdisc in enumerate(idiscs):
            result[i, j+1, 0] = 1
            result[i, j+1, 1] = _map_range(0, WIDTH, -1, 1, jdisc.pos.x)
            result[i, j+1, 2] = _map_range(0, WIDTH, -1, 1, jdisc.pos.y)
            result[i, j+1, 3] = jdisc.vel.x / DISC_SPEED
            result[i, j+1, 4] = jdisc.vel.y / DISC_SPEED
            result[i, j+1, 5] = _map_range(0, BOUNCES_INITIAL, -1, 1, jdisc.rem_bounces, True)
    return result

def _ricochet_from_agent_action(_raw_action, max_discs):
    raw_action = np.zeros((9,), dtype=np.float32)
    raw_action[1:9] = _raw_action
    raw_action = raw_action.reshape((3, 3))
    movement = vector(raw_action[0, 1], raw_action[0, 2]) * (MOVE_SPEED * 1.05)
    throw_discs = []
    for i in range(1, 3):
        if len(throw_discs) >= max_discs:
            break
        if raw_action[i, 0] <= 0:
            continue
        throw_dir = vector(raw_action[i, 1], raw_action[i, 2])
        throw_discs.append(throw_dir)
    return tick_action(movement, throw_discs)

def _ricochet_end_score(halfs, fails, by_error):
    """
    Compute the end of match score.
    """
    single_fail = sum(fails) == 1
    scores = [0, 0]
    for i in range(2):
        scores[1-i] = HP_INITIAL if fails[i] and by_error else HP_INITIAL - max(0, halfs[i].hero.hp)
    if single_fail:
        for i in range(2):
            scores[1-i] += fails[i] * MATCH_WIN_BONUS
    return scores[0] - scores[1]

def ricochet_match_py(opponent, xreward):
    """
    Generator that yields the current observation etc and then expects the action.
    """
    halfs = (half_state(hero(vector(1, 1)), []), half_state(hero(vector(WIDTH-1, HEIGHT-1)), []))
    last_score = 0
    tick_counter = 0
    reward = 0
    done = False
    info = {}
    if isinstance(opponent, base_bot):
        opponent.reset(True)
    while True:
        fails = [False] * 2
        # get charge
        for i in range(2):
            halfs[i].hero.charge = min(CHARGE_MAX, halfs[i].hero.charge + 1)
        # query bot
        actions = [None] * 2
        raw_actions = []
        obs = _ricochet_to_agent_observation(halfs)
        raw_action_0 = yield (obs, reward, done, info)
        if done:
            return
        if xreward:
            xreward_value = xreward.evaluate(obs, raw_action_0)
            reward += xreward_value
        raw_actions.append(raw_action_0)
        if not isinstance(opponent, base_bot):
            raw_action_1, _ = opponent.predict(obs[::-1])
            raw_actions.append(raw_action_1)
        actions = [_ricochet_from_agent_action(raw_actions[i], halfs[i].hero.charge // DISC_COST) for i in range(len(raw_actions))]
        if isinstance(opponent, base_bot):
            action_1 = opponent.get_action(halfs[1].copy(), halfs[0].copy())
            actions.append(action_1)
        for i in range(2):
            halfs[i].hero.charge -= DISC_COST * len(actions[i].throw_discs)
        # throw disc
        for i in range(2):
            for ivel in actions[i].throw_discs:
                halfs[1-i].discs.append(disc(next(global_id_generator), halfs[i].hero.pos, ivel))
        # do movement / do damage
        for i in range(2):
            halfs[i].update(actions[i].move_dir)
        # end?
        fails = [halfs[i].hero.hp <= 0 for i in range(2)]
        tick_counter += 1
        if any(fails) or tick_counter >= MATCH_MAX_TICKS:
            new_score = _ricochet_end_score(halfs, fails, False)
            done = True
        else:
            new_score = halfs[0].hero.hp - halfs[1].hero.hp
            done = False
        reward = new_score - last_score
        last_score = new_score

def ricochet_match_cpp(opponent):
    """
    Generator to run a match, with all the number crunching happening in low level C++

    Most of the runtime happens in the RL agents anyway, but running the game logic in C++
    as much as possible cuts down on the overhead. Overall runtime reduces
    by about 5% apparently, so it does help.
    Is it worth it? Well, as long as it's correct, it should be fine...

    Note: seems to possibly have a memory leak. System freezes up after a while in the session.
    Too risky. For now I use the Python version and accept the minor slowdown.
    """
    import ricochet_hetaro_accel
    flip = bool(random.randrange(2))
    fbuffer = ricochet_hetaro_accel.new_match()
    assert fbuffer[2] == 1000, str(fbuffer)
    while True:
        reward = fbuffer[0]
        done = fbuffer[1] > 0
        obs = np.array(fbuffer[3:157], dtype=np.float32).reshape((2, 11, 7))
        if flip:
            obs = obs[::-1]
            reward = -reward
        info = {}
        # query bot
        actions = [None] * 2
        raw_action_0 = yield (obs, reward, done, info)
        if done:
            return
        raw_action_1, _ = opponent.predict(obs[::-1])
        if flip:
            raw_action_0[[True, True, False, True, True, False, True, True]] *= -1
            raw_action_1[[True, True, False, True, True, False, True, True]] *= -1
        fbuffer[157:165] = raw_action_0
        fbuffer[165:173] = raw_action_1
        # update
        fbuffer = ricochet_hetaro_accel.update(fbuffer)
        assert fbuffer[2] < 1000, str(fbuffer)

ricochet_match = ricochet_match_py

def get_reference_action(reference: base_bot, match: match_state) -> np.ndarray:
    """
    Get the reference agent's action, in ML action format.
    """
    action = reference.get_action(match.halfs[0].copy(), match.halfs[1].copy())
    raw_action = np.zeros((8,), dtype=np.float32)
    raw_action[0] = action.move_dir.x / (MOVE_SPEED * 1.05)
    raw_action[1] = action.move_dir.y / (MOVE_SPEED * 1.05)
    raw_action[2] = raw_action[5] = -1
    for i, disc_dir in action.throw_discs:
        raw_action[i*3+2] = 1
        raw_action[i*3+3] = disc_dir.x
        raw_action[i*3+4] = disc_dir.y
    return raw_action

def get_reference_delta(ref_action: np.ndarray, our_action: np.ndarray) -> float:
    """
    Get a normalized reward value, which rewards something closer to the reference action more.
    """
    delta = 0
    delta += np.sum(np.abs(our_action[0:2] - ref_action[0:2])) * 2
    for i in range(2):
        if ref_action[i*3+2] > 0: # expect throw
            if our_action[i*3+2] > 0: # criticize aim
                delta += np.sum(np.abs(our_action[i*3+3:i*3+5] - ref_action[i*3+3:i*3+5]))
            else: # criticize no throw
                delta += 4 - 4 * our_action[i*3+2]
        else: # expect not throw
            if our_action[i*3+2] > 0: # criticize throw
                delta += 4 + 4 * our_action[i*3+2]
    reward = 8 - float(delta)
    return reward

class RicochetEnv(gym.Env):
    """
    Ricochet game. Agent to be trained/tested will face a randomly chosen
    opponent from the list of possible opponents. The opponent is static in
    the sense that it is not also learning in this game.
    """
    metadata = {'render.modes': ['human']}
    def __init__(self, opponents, xreward = None, reference = None):
        self._opponents = opponents
        self._xreward = None
        self._reference = None
        self._current_match = None
        # hero, disc1, disc2 | enable, vx, vy
        self.action_space = spaces.Box(low = -1, high = 1, shape = (8,), dtype = np.float32)
        # self, opponent | hero, disc1, disc2, ..., disc10 | exists, px, py, vx, vy, hp/bounces, charge
        self.observation_space = spaces.Box(low = -1, high = 1, shape = (2, 11, 7), dtype = np.float32)
    def step(self, action):
        obs, reward, done, info = self._current_match.send(action)
        if self._reference:
            reward = get_reference_delta(self._reference_action, action)
            self._reference_action = get_reference_action(self._reference, self._current_match)
        return obs, reward, done, info
    def reset(self):
        if self._xreward:
            self._xreward.reset()
        if self._reference:
            self._reference.reset(True)
        self._current_match = None
        opponent = random.choice(self._opponents)
        self._current_match = ricochet_match(opponent, self._xreward)
        obs, reward, done, info = next(self._current_match)
        if self._reference:
            self._reference_action = get_reference_action(self._reference, self._current_match)
        return obs
    def render(self, mode='human'):
        warnings.warn('RicochetEnv has no way to render it')
    def close(self):
        pass

_session_save_dir = None
_session_load_dir = None
def save_path(name, version, is_save):
    global _session_save_dir, _session_load_dir
    if is_save:
        _session_save_dir.mkdir(parents=True,exist_ok=True)
        return _session_save_dir / f'{name}_{version:>04}.zip'
    else:
        _session_load_dir.mkdir(parents=True,exist_ok=True)
        return _session_load_dir / f'{name}_{version:>04}.zip'
def save_parameters(agent, name, version, log=True):
    agent.save(save_path(name, version, True))
    if log:
        print('Just saved ' + name + ' version ' + str(version))
def load_parameters(env, name, version, log=True):
    lpath = save_path(name, version, False)
    lpath = lpath.with_suffix('')
    result = PPO.load(lpath, env)
    if log:
        print('Just loaded ' + name + ' version ' + str(version))
    return result
def load_latest(env, index, require = True):
    versions = []
    for fp in _session_load_dir.iterdir():
        if not fp.is_file():
            continue
        fn = fp.name
        match = re.match('([a-z]+(\\d+))_(\\d+).zip', fn)
        if not match:
            continue
        match_index = int(match[2])
        if match_index != index:
            continue
        version_number = int(match[3])
        agent_name = match[1]
        versions.append((version_number, agent_name))
    if versions:
        version_number, agent_name = max(versions)
        agent = load_parameters(env, agent_name, version_number)
        return agent, agent_name, version_number
    else:
        if require:
            raise ValueError(f'No file to load for agent #{index}')
        print('No existing agent #' + str(index) + ' to load')
        return None

def get_options():
    global _session_save_dir, _session_load_dir, _use_v3_model
    parser = argparse.ArgumentParser(description='Trains the RL agents for Hetaro.')
    parser.add_argument('-o', '--output', type=str, required=True,
        help='Directory to load/save final agent parameters to.')
    parser.add_argument('--fork', default=False, action='store_true',
        help='Randomly instantiate some new agents for the live group by copying the static group.\n' + \
             'If static group is empty, makes blank agents instead. Useful for bootstrapping.')
    parser.add_argument('--refine', default=False, action='store_true',
        help='Run through the live group, training them against each other and the static group.')
    parser.add_argument('--do-export', default=False, action='store_true',
        help='Exports the live group agent data in a portable format. Not guaranteed to be a perfect backup.')
    parser.add_argument('--do-import', default=False, action='store_true',
        help='Imports the live group agent data from a portable format. Not guaranteed to be a perfect backup.')
    parser.add_argument('--static', default=None, type=int, nargs=2,
        help='Start and stop range for the static group.')
    parser.add_argument('--live', default=None, type=int, nargs=2,
        help='Start and stop range for the live group.')
    parser.add_argument('--fork-prefix', default=None, type=str,
        help='Set a name prefix for the forks.')
    parser.add_argument('--xrewards', default=None, type=str, nargs='+',
        help='Extra rewards to give during training to incentivize an agent to pursue an additional goal.\n' + \
             'Supported xrewards:\n' + \
             '* near - Rewards the agent for being near the opponent.\n' + \
             '* far - Rewards the agent for being far from the opponent.\n' + \
             '* aim - Rewards the agent for aiming accurately.\n' + \
             '* dodge - Punishes (negative reward to) the agent when taking damage.\n' + \
             '* doubleshot - Rewards the agent for performing a double-shot.\n' + \
             '* manydisc - Rewards the agent for having many own discs out at once.\n' + \
             '* wastecharge - Punishes (negative reward to) the agent for being at max charge.\n' + \
             '* bouncehit - Reward the agent for hitting the opponent with a bounced disc.')
    parser.add_argument('--iteration-size', default=2*10**5, type=int,
        help='How many ticks per learning iteration.')
    parser.add_argument('--vs-reid', default=False, action='store_true',
        help='Add Reid 2 to the static opponents.')
    parser.add_argument('--vs-reid-2', default=False, action='store_true',
        help='Add Reid to the static opponents.')
    parser.add_argument('--big-model', default=False, action='store_true',
        help='Use the V3 model size when creating new agents. Only necessary during fork.')
    parser.add_argument('--reference-reid', default=False, action='store_true',
        help='Imitation learning: use Reid 2 as reference.')
    options = parser.parse_args()
    if (options.fork or options.refine or options.do_import or options.do_export) and not (options.static and options.live):
        raise ValueError('Static range or live range is missing')
    _session_load_dir = _session_save_dir = pathlib.Path(options.output)
    options.xreward_live = {}
    options.xopponents = []
    if options.xrewards:
        if not options.refine:
            raise ValueError('xreward can only be used with refine')
        if len(options.xrewards) > options.live[1] - options.live[0]:
            raise ValueError('Not enough live agents to assign xrewards to')
        xreward_live = []
        for xr_name in options.xrewards:
            if xr_name not in xreward_all:
                raise ValueError(f'Unknown xreward "{xr_name}"')
            xreward_live.append(xreward_all[xr_name]())
        xreward_live = dict(enumerate(xreward_live))
        options.xreward_live = xreward_live
    if options.vs_reid:
        options.xopponents.append(reid_bot('Reid', use_hunter=False, mc_expand=7, mc_iters=3, spread_multiplier=0.38))
    if options.vs_reid_2:
        options.xopponents.append(reid_bot('Reid 2'))
    if options.big_model:
        _use_v3_model = True
    options.reference = None
    if options.reference_reid:
        options.reference = reid_bot('Reid 2')
    return options

_time_last = None
def time_since_last():
    global _time_last
    time_now = datetime.datetime.now()
    if _time_last is None:
        _time_last = time_now
    result = time_now - _time_last
    _time_last = time_now
    return result

def expected_best(num_static, num_live):
    return num_static * 8 / (num_static + num_live - 1)

def report_evaluate(env, agent, tag, trials=50):
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=trials)
    print(f'Agent {tag}: {mean_reward} +/- {std_reward} (vs {len(env._opponents)} opponents | {time_since_last()} since last report)')

_use_v3_model = False
def get_model_spec():
    """
    The model spec to be used by our agents.

    Rationale (vs Reid):
    154 - Observation size. Practically 100 actual variables.
    128 - Extract useful features.
    256 - Compute trajectories (fused move + bounce).
    128 - Detect intersections (part 1).
    128 - Detect intersections (part 2).
    32 - MC action values.
    8 - Action size.

    This is roughly the amount of computation needed.
    From there, we spread it out and make the model deeper, and that gets the end net.

    Codename: compress and crunch
    """
    global _use_v3_model
    if _use_v3_model:
        return [256] + [128] * 3 + [64] * 4 + [32] * 4
    return [64] * 4 + [32] * 8

def make_blank_agent(env):
    """
    Make a new blank agent, with some overrides and the rest default settings.
    """
    policy_kwargs = {'net_arch': get_model_spec()}
    return PPO('MlpPolicy', env, policy_kwargs = policy_kwargs)

def main():
    options = get_options()
    print('Let\'s make Hetaro!')
    if options.fork:
        phase_fork(options.static, options.live, options.fork_prefix or 'forked')
    if options.do_export:
        phase_export(options.live)
    if options.do_import:
        phase_import(options.live)
    if options.refine:
        phase_refine(options.static, options.live, options.xreward_live, options.xopponents, options.iteration_size, options.reference)

def phase_fork(static, live, fork_prefix):
    env = RicochetEnv([])
    print('='*40)
    print(f'Forking {static} -> {live}')
    static_start, static_stop = static
    live_start, live_stop = live
    live_group = []
    while len(live_group) < live_stop - live_start:
        if static_stop - static_start == 0:
            agent = make_blank_agent(env)
            live_group.append(agent)
            continue
        cut = min(static_stop - static_start, live_stop - live_start - len(live_group))
        indices = random.sample(range(static_start, static_stop), cut)
        for index in indices:
            agent, agent_name, version_number = load_latest(env, index)
            live_group.append(agent)
    random.shuffle(live_group)
    for index, agent in zip(range(live_start, live_stop), live_group):
        save_parameters(agent, fork_prefix + str(index), 0)

def phase_refine(static, live, xreward_live, xopponents, iteration_size, reference):
    env = RicochetEnv([], reference=reference)
    print('='*40)
    print(f'Refining {static} -> {live}')
    static_start, static_stop = static
    live_start, live_stop = live
    print('Optimal average reward: ' + str(expected_best(static_stop - static_start + len(xopponents), live_stop - live_start)))
    static_group = []
    live_group = []
    for index in range(static_start, static_stop):
        agent, agent_name, version_number = load_latest(env, index)
        static_group.append((agent, agent_name, version_number))
    static_group += [(agent, agent.get_bot_name(), 0) for agent in xopponents]
    for index in range(live_start, live_stop):
        agent, agent_name, version_number = load_latest(env, index)
        live_group.append((agent, agent_name, version_number))
    print('Establish baseline performance')
    for j in range(len(live_group)):
        agent, agent_name, version_number = live_group[j]
        env._opponents = [atup[0] for atup in static_group + live_group[:j] + live_group[j+1:]]
        report_evaluate(env, agent, agent_name)
    print('Beginning refine loop')
    time_since_last()
    while True:
        for j in random.sample(range(len(live_group)), len(live_group)):
            print(f'Current time is {datetime.datetime.utcnow()}')
            agent, agent_name, version_number = live_group[j]
            version_number += 1
            env._opponents = [atup[0] for atup in static_group + live_group[:j] + live_group[j+1:]]
            env._xreward = xreward_live.get(j, None)
            agent.learn(total_timesteps = iteration_size)
            env._xreward = None
            live_group[j] = agent, agent_name, version_number
            report_evaluate(env, agent, agent_name)
            save_parameters(agent, agent_name, version_number)

def phase_import(live):
    env = RicochetEnv([])
    print('='*40)
    print(f'Importing all matching in {live}')
    live_start, live_stop = live
    for fp in _session_load_dir.iterdir():
        if not fp.is_file():
            continue
        fn = fp.name
        match = re.match('([a-z]+(\\d+))_(\\d+).pt', fn)
        if not match:
            continue
        match_index = int(match[2])
        if not live_start <= match_index < live_stop:
            continue
        version_number = int(match[3])
        agent_name = match[1]
        # import it
        agent = make_blank_agent(env)
        params = torch.load(fp)
        agent.set_parameters(params)
        save_parameters(agent, agent_name, version_number)

def phase_export(live):
    env = RicochetEnv([])
    print('='*40)
    print(f'Exporting latest of each agent in {live}')
    live_start, live_stop = live
    for index in range(live_start, live_stop):
        agent, agent_name, version_number = load_latest(env, index)
        params = agent.get_parameters()
        torch.save(params, _session_save_dir / f'{agent_name}_{version_number:>04}.pt')

if __name__ == '__main__':
    main()
