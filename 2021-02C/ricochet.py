"""
18x12 continuous field (effectively 17x11).
Player heroes cannot enter the outer 1 edge, so bounds are 15x9 to the hero.
2 players compete in a match.
Players move around the field and throw discs to damage the opponent.
Players and discs have radius 0.5.
Players can be hit 5 times before they lose the match.
A disc can hit a player multiple times.
Discs can bounce off the wall 4 times. They are absorbed the 5th time.
Players can move 0.25 units per tick.
Discs move 0.5 units per tick.
Players gain 1 disc charge per tick.

Tick order:
* Players gain disc charge. They can hold up to 20 charge. 10 charge is needed to throw a disc.
* Players issue orders, which may include movement and throwing discs.
* Players throw discs.
* Players and discs do movement. Discs do ricochet. Expired discs are removed.
* Discs do damage.

Matches last up to 1000 ticks before a draw is called.
Players gain 1 point per damage dealt, plus 3 bonus points if they win the match.
"""

import abc
import collections
import itertools
import math
import traceback
import typing

class vector(object):
    """
    Simple 2D vector class.
    """
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    def assert_finite(self):
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise AssertionError(f'Vector components are not finite: {self.x}, {self.y}')
    def __str__(self):
        return f'({self.x:.4g}, {self.y:.4g})'
    def __repr__(self):
        return 'vector(' + repr(self.x) + ', ' + repr(self.y) + ')'
    def __eq__(self, other: typing.Any) -> bool:
        return self is other or isinstance(other, vector) and self.x == other.x and self.y == other.y
    def __ne__(self, other: typing.Any) -> bool:
        return not (self == other)
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    def __add__(self, other):
        return vector(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return vector(self.x - other.x, self.y - other.y)
    def __mul__(self, other: float):
        return vector(self.x * other, self.y * other)
    def __truediv__(self, other: float):
        return vector(self.x / other, self.y / other)
    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)
    def distance(self, other) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)
    def normalize(self):
        m = self.magnitude()
        return self / m if m else self
    def copy(self):
        return vector(self.x, self.y)

MOVE_SPEED = 0.25
DISC_SPEED = 0.5
DISC_COST = 10
CHARGE_MAX = 20
HP_INITIAL = 5
BOUNCES_INITIAL = 4
INTERSECT_DISTANCE = 1.0
WIDTH = 17
HEIGHT = 11

MATCH_WIN_BONUS = 3
MATCH_MAX_TICKS = 1000
TOURNAMENT_OUTER_ITERATIONS = 10 # subject to change
TOURNAMENT_MATCHES = 100 # subject to change

class tick_action(object):
    """
    Represents an action issued by a bot player.
    Note that movement vectors are clamped to the max movement speed,
    and disc vectors are normalized to the disc speed.
    """
    def __init__(self, move_dir: vector, throw_discs: typing.List[vector]):
        move_dir_mag = move_dir.magnitude()
        if move_dir_mag > MOVE_SPEED:
            move_dir = move_dir * (MOVE_SPEED / move_dir_mag)
        self.move_dir = move_dir
        throw_discs = [iv * (DISC_SPEED / iv.magnitude()) for iv in throw_discs]
        self.throw_discs = throw_discs
        move_dir.assert_finite()
        for iv in throw_discs:
            iv.assert_finite()

class base_bot(abc.ABC):
    @abc.abstractmethod
    def get_bot_name(self) -> str:
        pass
    @abc.abstractmethod
    def get_player_name(self) -> str:
        pass
    @abc.abstractmethod
    def reset(self, got_new_opponent: bool):
        pass
    @abc.abstractmethod
    def get_action(self, my_half_state, opp_half_state) -> tick_action:
        pass
    def __str__(self) -> str:
        return self.get_bot_name()

class hero(object):
    """
    Represents the hero character.
    """
    def __init__(self, pos: vector, hp: int = HP_INITIAL, charge: int = 0):
        self.pos = pos
        self.hp = hp
        self.charge = charge
    def clamp(self):
        self.pos = vector(min(WIDTH-1, max(1, self.pos.x)), min(HEIGHT-1, max(1, self.pos.y)))
    def copy(self):
        return hero(self.pos, self.hp, self.charge)

class disc(object):
    """
    Represents a single disc.
    """
    def __init__(self, id: int, pos: vector, vel: vector, rem_bounces: int = BOUNCES_INITIAL):
        self.id = id
        self.pos = pos
        self.vel = vel
        self.rem_bounces = rem_bounces
    def clamp(self):
        if self.pos.x < 0:
            self.pos.x = -self.pos.x
            self.vel.x = -self.vel.x
            self.rem_bounces -= 1
        if self.pos.x > WIDTH:
            self.pos.x = 2*WIDTH-self.pos.x
            self.vel.x = -self.vel.x
            self.rem_bounces -= 1
        if self.pos.y < 0:
            self.pos.y = -self.pos.y
            self.vel.y = -self.vel.y
            self.rem_bounces -= 1
        if self.pos.y > HEIGHT:
            self.pos.y = 2*HEIGHT-self.pos.y
            self.vel.y = -self.vel.y
            self.rem_bounces -= 1
    def copy(self):
        return disc(self.id, self.pos, self.vel, self.rem_bounces)

def _id_generator():
    import random
    orandom = random.Random()
    DUMP_AT = 100
    CUT_AT = 50
    buffer = []
    for i in itertools.count():
        buffer.append(i)
        if len(buffer) >= DUMP_AT:
            orandom.shuffle(buffer)
            yield from buffer[CUT_AT:]
            buffer = buffer[:CUT_AT]
global_id_generator = iter(_id_generator())

class half_state(object):
    """
    Represents half of the game state.
    """
    def __init__(self, hero: hero, discs: typing.List[disc]):
        self.hero = hero
        self.discs = discs
    def update(self, hero_vel: vector):
        """
        Perform a single game tick.
        """
        hero = self.hero
        discs = self.discs
        # all do movement
        hero.pos = hero.pos + hero_vel
        hero.clamp()
        for i in range(len(discs))[::-1]:
            disc = discs[i]
            disc.pos = disc.pos + disc.vel
            disc.clamp()
            if disc.rem_bounces < 0:
                del discs[i]
        # do damage
        for disc in discs:
            if hero.pos.distance(disc.pos) < INTERSECT_DISTANCE:
                hero.hp -= 1
    def copy(self):
        return half_state(self.hero.copy(), list(map(disc.copy, self.discs)))

class match_state(object):
    """
    Represents the state of the match.
    """
    def __init__(self, players: typing.Tuple[base_bot, base_bot], halfs: typing.Optional[typing.Tuple[half_state, half_state]] = None, tick_counter: int = 0):
        if halfs is None:
            halfs = (half_state(hero(vector(1, 1)), []), half_state(hero(vector(WIDTH-1, HEIGHT-1)), []))
        self.players = players
        self.halfs = halfs
        self.tick_counter = tick_counter
    def end_score(self, fails: typing.Tuple[bool, bool], by_error: bool) -> typing.Tuple[int, int]:
        """
        Compute the end of match score.
        """
        single_fail = sum(fails) == 1
        scores = [0, 0]
        for i in range(2):
            scores[1-i] = HP_INITIAL if fails[i] and by_error else HP_INITIAL - max(0, self.halfs[i].hero.hp)
        if single_fail:
            for i in range(2):
                scores[1-i] += fails[i] * MATCH_WIN_BONUS
        return tuple(scores)
    def update(self) -> typing.Optional[typing.Tuple[int, int]]:
        """
        Perform one game tick. If the game ended, return the scores.
        """
        fails = [False] * 2
        # get charge
        for i in range(2):
            self.halfs[i].hero.charge = min(CHARGE_MAX, self.halfs[i].hero.charge + 1)
        # query bot
        actions = [None] * 2
        for i in range(2):
            try:
                actions[i] = self.players[i].get_action(self.halfs[i].copy(), self.halfs[1-i].copy())
            except Exception as exc:
                traceback.print_exc()
                fails[i] = True
        if any(fails):
            return self.end_score(fails, True)
        for i in range(2):
            self.halfs[i].hero.charge -= DISC_COST * len(actions[i].throw_discs)
            if self.halfs[i].hero.charge < 0:
                fails[i] = True
        # throw disc
        if any(fails):
            return self.end_score(fails, True)
        for i in range(2):
            for ivel in actions[i].throw_discs:
                self.halfs[1-i].discs.append(disc(next(global_id_generator), self.halfs[i].hero.pos, ivel))
        # do movement / do damage
        for i in range(2):
            self.halfs[i].update(actions[i].move_dir)
        # end?
        fails = [self.halfs[i].hero.hp <= 0 for i in range(2)]
        self.tick_counter += 1
        if any(fails) or self.tick_counter >= MATCH_MAX_TICKS:
            return self.end_score(fails, False)

# BEGIN --- define bots below!

class random_bot(base_bot):
    """
    Some basic bots to fill space.
    """
    def __init__(self, use_movement: bool, use_aiming: bool):
        import random
        self._random = random.Random()
        self._use_movement = use_movement
        self._use_aiming = use_aiming
    def get_bot_name(self):
        return [['random_stuck_bot','basedball_bot'],['random_bot','example_bot']][self._use_movement][self._use_aiming]
    def get_player_name(self):
        return 'examples'
    def reset(self, no):
        pass
    def get_action(self, shalf, ohalf):
        random = self._random
        if self._use_movement:
            ang = random.uniform(-math.pi, math.pi)
            movement = vector(math.cos(ang), math.sin(ang)) * 100
        else:
            movement = vector(0, 0)
        throw_discs = []
        if shalf.hero.charge >= DISC_COST:
            if self._use_aiming:
                throw_dir = ohalf.hero.pos - shalf.hero.pos
            else:
                ang = random.uniform(-math.pi, math.pi)
                throw_dir = vector(math.cos(ang), math.sin(ang)) * 100
            throw_discs.append(throw_dir)
        return tick_action(movement, throw_discs)

class melee_bot(base_bot):
    """
    Ranged weapon? What ranged weapon?
    """
    def get_bot_name(self):
        return 'melee_bot'
    def get_player_name(self):
        return 'examples'
    def reset(self, no):
        pass
    def get_action(self, shalf, ohalf):
        DUMP_THRESHOLD = INTERSECT_DISTANCE
        WAIT_THRESHOLD = DUMP_THRESHOLD + MOVE_SPEED * DISC_COST / 2
        delta = ohalf.hero.pos - shalf.hero.pos
        delta_mag = delta.magnitude()
        movement = delta * (100 / delta_mag)
        num_discs_to_throw = 0
        if shalf.hero.charge >= CHARGE_MAX and delta_mag > WAIT_THRESHOLD:
            num_discs_to_throw = 1
        if delta_mag < DUMP_THRESHOLD:
            num_discs_to_throw = shalf.hero.charge // DISC_COST
        throw_discs = [delta] * num_discs_to_throw
        return tick_action(movement, throw_discs)

class reid_bot(base_bot):
    """
    Monte Carlo based bot.
    Uses next game state prediction to dodge discs.
    Always throws 2 discs at once in a spread pattern.
    """
    def __init__(self, use_eo: bool):
        self._use_eo = use_eo
    def get_bot_name(self):
        result = 'Reid'
        if self._use_eo:
            result += '+EggMod'
        return result
    def get_player_name(self):
        return 'komiamiko'
    def reset(self, no):
        import random
        self._random = random.Random()
        self._eo_toggle = False
    def get_action(self, shalf, ohalf):
        random = self._random
        best_movement, _ = self._mc_route(shalf, 7, 3, 1, 0.9)
        throw_discs = []
        if shalf.hero.charge >= 2 * DISC_COST:
            mypos = shalf.hero.pos
            target = ohalf.hero.pos.copy()
            if self._use_eo:
                self._eo_toggle = not self._eo_toggle
            if self._eo_toggle:
                # go for a bounce shot
                if random.randrange(3) == 0:
                    pass
                elif (mypos.x < WIDTH / 2) ^ (random.randrange(3) == 0):
                    target.x = -target.x
                else:
                    target.x = 2*WIDTH-target.x
                if random.randrange(3) == 0:
                    pass
                elif (mypos.y < HEIGHT / 2) ^ (random.randrange(3) == 0):
                    target.y = -target.y
                else:
                    target.y = 2*HEIGHT-target.y
            delta = target - mypos
            delta_mag = delta.magnitude()
            project_at = max(INTERSECT_DISTANCE * 2, delta_mag)
            spread = 0.35 / project_at
            throw_discs.append(vector(delta.x + delta.y * spread, delta.y - delta.x * spread))
            throw_discs.append(vector(delta.x - delta.y * spread, delta.y + delta.x * spread))
        return tick_action(best_movement, throw_discs)
    def _mc_route(self, shalf: half_state, expand_to: int, extend: int, rem_branches: int, future_decay: float) -> typing.Tuple[vector, float]:
        """
        Evaluate some possible futures, and return what we think will get the best result.
        """
        random = self._random
        best_movement = vector(0, 0)
        best_score = -1e50
        movement_candidates = [vector(0, 0)]
        for _ in range(expand_to - 1):
            ang = random.uniform(-math.pi, math.pi)
            movement_candidates.append(vector(math.cos(ang), math.sin(ang)) * MOVE_SPEED)
        for imovement in movement_candidates:
            ivalue = 0
            ishalf = shalf.copy()
            decay_fac = 1
            for _ in range(extend):
                hp_before = ishalf.hero.hp
                ishalf.update(imovement)
                hp_after = ishalf.hero.hp
                ivalue += (hp_after - hp_before) * decay_fac
                decay_fac *= future_decay
            if rem_branches > 0:
                _, cvalue = self._mc_route(ishalf, expand_to, extend, rem_branches - 1, future_decay)
                ivalue += decay_fac * cvalue
            if ivalue > best_score:
                best_movement = imovement
                best_score = ivalue
        return best_movement, best_score

# END --- define bots above!

def run_match(p0: base_bot, p1: base_bot) -> typing.Tuple[int, int]:
    state = match_state((p0, p1))
    while True:
        res = state.update()
        if res is not None:
            return res

def make_all_bots() -> typing.List[base_bot]:
    return [
        random_bot(False, False),
        random_bot(False, True),
        random_bot(True, False),
        random_bot(True, True),
        melee_bot(),
        reid_bot(False),
        ]

def run_tournament(all_bots: typing.List[base_bot]) -> typing.List[typing.Tuple[int, str, str]]:
    import random
    orandom = random.Random()
    score_counters = collections.Counter()
    all_pairs = list(itertools.combinations(all_bots, 2))
    for ii in range(TOURNAMENT_OUTER_ITERATIONS):
        orandom.shuffle(all_pairs)
        for p0, p1 in all_pairs:
            p0.reset(True)
            p1.reset(True)
            c0 = c1 = 0
            for jj in range(TOURNAMENT_MATCHES):
                if orandom.randrange(2) == 0:
                    p0, p1 = p1, p0
                    c0, c1 = c1, c0
                d0, d1 = run_match(p0, p1)
                c0 += d0
                c1 += d1
            if ii == 0:
                print(f'{p0} vs {p1} - {c0}:{c1}')
            score_counters[str(p0)] += c0
            score_counters[str(p1)] += c1
        print(f'big iteration {ii+1} / {TOURNAMENT_OUTER_ITERATIONS}')
    score_table = [(score_counters[str(p0)], str(p0), p0.get_player_name()) for p0 in all_bots]
    score_table.sort(reverse=True)
    return score_table

def print_score_table(score_table: typing.List[typing.Tuple[int, str, str]]):
    print('\n'.join(f'# {tup[0]:>7} | {tup[1]:>20} by {tup[2]}' for tup in score_table))

def main():
    all_bots = make_all_bots()
    score_table = run_tournament(all_bots)
    print_score_table(score_table)

if __name__ == '__main__':
    main()