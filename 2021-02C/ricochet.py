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
import argparse
import collections
import datetime
import functools
import itertools
import json
import math
import pathlib
import subprocess
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
    def dot(self, other):
        return self.x * other.x + self.y + other.y
    def cosine(self, other) -> float:
        return self.dot(other) / (self.magnitude() * other.magnitude() or 1)
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
TOURNAMENT_PATIENCE = datetime.timedelta(seconds=10) # subject to change

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
        return hero(self.pos.copy(), self.hp, self.charge)

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
        return disc(self.id, self.pos.copy(), self.vel.copy(), self.rem_bounces)

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

@functools.lru_cache
def record_random_height(disc_id: int) -> float:
    """
    Get the pseudorandom height for the disc.
    """
    xx = 1 / (1 + disc_id)
    for _ in range(10):
        xx = xx * (1 - xx) * 3.59
    return 0.8 + xx * 0.15

def record_is_linear(v1: typing.Tuple[float, float, float], v2: typing.Tuple[float, float, float], v3: typing.Tuple[float, float, float]):
    """
    Are these 3 points (almost) following a linear recurrence?
    """
    delta = sum((v1[i] - 2 * v2[i] + v3[i]) ** 2 for i in range(3))
    return delta < 1e-7

class recorded_half_state(half_state):
    """
    Like half_state except it records the replay data.
    """
    def __init__(self, hero: hero, discs: typing.List[disc]):
        half_state.__init__(self, hero, discs)
        self._last_move_dir = None
    def update(self, ref_match: typing.Any, aframe: int, jj: int, move_dir: typing.Optional[vector]):
        """
        Perform a fractional game tick.
        """
        hero = self.hero
        discs = self.discs
        phase = aframe % 4
        hero_vel = self._last_move_dir = move_dir or self._last_move_dir
        # all do movement
        hero.pos = hero.pos + hero_vel * (1/4)
        hero.clamp()
        ref_match.frames[f'Hero{jj}:location'][aframe] = (hero.pos.x, hero.pos.y, 0.85)
        for i in range(len(discs))[::-1]:
            disc = discs[i]
            disc.pos = disc.pos + disc.vel * (1/4)
            disc.clamp()
            if disc.rem_bounces < 0:
                ref_match.frames[f'Disc{disc.id}:location'][aframe] = (disc.pos.x, disc.pos.y, 0)
                del discs[i]
            else:
                ref_match.frames[f'Disc{disc.id}:location'][aframe] = (disc.pos.x, disc.pos.y, record_random_height(disc.id))
        # do damage
        if phase == 3:
            hp_before = hero.hp
            for disc in discs:
                if hero.pos.distance(disc.pos) < INTERSECT_DISTANCE:
                    hero.hp -= 1
            hp_after = hero.hp
            if hp_after < hp_before:
                for k in range(1, 6):
                    if hp_before >= k and hp_after < k:
                        ref_match.frames[f'Gem{jj}{k}:scale'][aframe-1] = (0.45,) * 3
                        ref_match.frames[f'Gem{jj}{k}:scale'][aframe] = (0.15,) * 3
    def copy(self):
        """
        Make a copy which is a regular half_state instead of a recorded_half_state
        """
        return half_state(self.hero.copy(), list(map(disc.copy, self.discs)))

class recorded_match_state(match_state):
    """
    Like match_state except it records the replay data.
    """
    def __init__(self, players: typing.Tuple[base_bot, base_bot]):
        halfs = (recorded_half_state(hero(vector(1, 1)), []), recorded_half_state(hero(vector(WIDTH-1, HEIGHT-1)), []))
        self.players = players
        self.halfs = halfs
        self.tick_counter = 0
        self.subtick = 0
        self.frames = collections.defaultdict(dict)
        self.discs = [set(), set()]
        self.match_result = None
    def update(self) -> typing.Optional[typing.Tuple[int, int]]:
        """
        Perform one game tick. If the game ended, return the scores.
        """
        self.subtick = (self.subtick + 1) % 4
        aframe = int(round(self.tick_counter * 4)) + 4
        phase = aframe % 4
        fails = [False] * 2
        # get charge
        for i in range(2):
            self.halfs[i].hero.charge = min(CHARGE_MAX, self.halfs[i].hero.charge + 1/4)
        if phase == 0:
            for i in range(2):
                self.halfs[i].hero.charge = int(self.halfs[i].hero.charge)
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
                charge_before = self.halfs[i].hero.charge
                charge_after = self.halfs[i].hero.charge = charge_before - DISC_COST * len(actions[i].throw_discs)
                if self.halfs[i].hero.charge < 0:
                    fails[i] = True
                if charge_after < charge_before:
                    self.frames[f'Charge{i}:scale'][aframe-1] = (1, charge_before / CHARGE_MAX, 1)
                self.frames[f'Charge{i}:scale'][aframe] = (1, charge_after / CHARGE_MAX, 1)
            # throw disc
            if any(fails):
                return self.end_score(fails, True)
            for i in range(2):
                for ivel in actions[i].throw_discs:
                    new_disc = disc(next(global_id_generator), self.halfs[i].hero.pos, ivel)
                    self.halfs[1-i].discs.append(new_disc)
                    disc_id = new_disc.id
                    self.discs[i].add(disc_id)
                    self.frames[f'Disc{disc_id}:location'][aframe-1] = (self.halfs[i].hero.pos.x, self.halfs[i].hero.pos.y, 0)
                    self.frames[f'Disc{disc_id}:location'][aframe] = (self.halfs[i].hero.pos.x, self.halfs[i].hero.pos.y, record_random_height(disc_id))
        # do movement / do damage
        for i in range(2):
            self.halfs[i].update(self, aframe, i, (actions[i].move_dir if phase == 0 else None))
        # end?
        fails = [self.halfs[i].hero.hp <= 0 for i in range(2)]
        self.tick_counter += 1/4
        if any(fails) or self.tick_counter >= MATCH_MAX_TICKS:
            return self.end_score(fails, False)
    def export(self, match_index: int, num_matches: int, score0: int, score1: int):
        """
        Export the animation file for this match.
        """
        lines = []
        lines.append('import bpy')
        lines.append('scn = bpy.context.scene')
        # set ending text
        ds0, ds1 = self.match_result
        score0 += ds0
        score1 += ds1
        lines.append(f'bpy.data.objects["ResultText"].data.body = "{self.players[0]} vs {self.players[1]}\\n{score0} : {score1}"')
        # make any final keyframes
        afirst = min(map(min, self.frames.values()))
        alast = max(map(max, self.frames.values()))
        self.frames['ResultText:location'][alast] = (WIDTH/2, HEIGHT/2, 0)
        self.frames['ResultText:location'][alast+1] = (WIDTH/2, HEIGHT/2, 1.5)
        # set animation start/end
        lines.append(f'scn.frame_start = {afirst}')
        lines.append(f'scn.frame_end = {alast+40}')
        # duplicate discs
        lines.append('def CN(src_name, new_name):')
        lines.append('  src_obj = bpy.data.objects[src_name]')
        lines.append('  new_obj = src_obj.copy()')
        lines.append('  new_obj.data = src_obj.data.copy()')
        lines.append('  new_obj.animation_data.action = src_obj.animation_data.action.copy()')
        lines.append('  new_obj.name = new_name')
        lines.append('  bpy.context.collection.objects.link(new_obj)')
        lines.append('')
        for i in range(2):
            for disc_id in self.discs[i]:
                lines.append(f'CN("DiscRef{i}", "Disc{disc_id}")')
        # optimize movement keyframes
        for kfpack, kfs in self.frames.items():
            name, prop = kfpack.split(':')
            if prop == 'location' or prop == 'scale':
                rmset = set()
                kfskeys = set(kfs)
                for t in range(afirst, alast):
                    if set(range(t,t+3)) <= kfskeys and \
                        record_is_linear(kfs[t  ], kfs[t+1], kfs[t+2]):
                        rmset.add(t+1)
                for t in rmset:
                    del kfs[t]
        # add all keyframes
        for kfpack, kfs in self.frames.items():
            name, prop = kfpack.split(':')
            lines.append(f'cc = bpy.data.objects["{name}"]')
            lines.append(f'kv = {{ {", ".join(str(t)+":"+str(v) for t,v in kfs.items())} }}')
            lines.append('for t, v in kv.items():')
            lines.append(f'  cc.{prop} = v')
            lines.append(f'  cc.keyframe_insert(data_path="{prop}", frame=t)')
            lines.append('')
        lines.append(f'for oname in {[kfpack.split(":")[0] for kfpack in self.frames]}:')
        lines.append('  cc = bpy.data.objects[oname]')
        lines.append('  fcurves = cc.animation_data.action.fcurves')
        lines.append('  for fcurve in fcurves:')
        lines.append('    for kf in fcurve.keyframe_points:')
        lines.append('      kf.interpolation = "LINEAR"')
        lines.append('')
        # write to file
        lines.append('')
        raw = '\n'.join(lines)
        print(f'Generated {len(lines)} lines = {len(raw) // 1024} KiB of animation data')
        out_file = pathlib.Path('visualize', 'replay')
        out_file.mkdir(parents=True, exist_ok=True)
        with open(out_file / f'{match_index+1:>04}.py', 'w') as file:
            file.write(raw)

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
    def __init__(self, name: str, use_eo: bool = False, use_hunter: bool = True,
        mc_expand: int = 8, mc_iters: int = 5, spread_multiplier: float = 0.75):
        self._name = name
        self._use_eo = use_eo
        self._use_hunter = use_hunter
        self._mc_expand = mc_expand
        self._mc_iters = mc_iters
        self._spread_multiplier = spread_multiplier
    def get_bot_name(self):
        return self._name
    def get_player_name(self):
        return 'komiamiko'
    def reset(self, no):
        import random
        self._random = random.Random()
        self._eo_toggle = False
    def get_action(self, shalf, ohalf):
        random = self._random
        best_movement, _ = self._mc_route(shalf, ohalf, self._mc_expand, self._mc_iters, 1, 0.8)
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
            SPREAD_FACTOR = math.sqrt(1 - (MOVE_SPEED + DISC_SPEED) / (2 * INTERSECT_DISTANCE)) * self._spread_multiplier
            spread = SPREAD_FACTOR * INTERSECT_DISTANCE / project_at
            throw_discs.append(vector(delta.x + delta.y * spread, delta.y - delta.x * spread))
            throw_discs.append(vector(delta.x - delta.y * spread, delta.y + delta.x * spread))
        return tick_action(best_movement, throw_discs)
    def _mc_route(self, shalf: half_state, ohalf: half_state, expand_to: int, extend: int, rem_branches: int, future_decay: float) -> typing.Tuple[vector, float]:
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
                jvalue = (hp_after - hp_before)
                if self._use_hunter:
                    IDEAL_DISTANCE = INTERSECT_DISTANCE * 2.1 / MOVE_SPEED * DISC_SPEED
                    jvalue -= 0.01 * abs(ishalf.hero.pos.distance(ohalf.hero.pos))
                jvalue *= decay_fac
                ivalue += jvalue
                decay_fac *= future_decay
            if rem_branches > 0:
                _, cvalue = self._mc_route(ishalf, ohalf, expand_to, extend, rem_branches - 1, future_decay)
                ivalue += decay_fac * cvalue
            if ivalue > best_score:
                best_movement = imovement
                best_score = ivalue
        return best_movement, best_score

class tj_bot(base_bot):
    def __init__(self):
        import random
        self._random = random.Random()
    def get_bot_name(self):
        return 'Tj Bot'
    def get_player_name(self):
        return 'tj'
    def reset(self, no):
        self.throw_next = False
        import tj
        tj.reset(no)
        self._use_movement = True
        self._use_aiming = True
        pass
    def get_action(self, shalf, ohalf):
        random = self._random
        import tj
        discs = []
        for d in shalf.discs:
            discs.append((d.pos.x, d.pos.y, d.vel.x, d.vel.y, d.rem_bounces))
        for d in ohalf.discs:
            discs.append((d.pos.x, d.pos.y, d.vel.x, d.vel.y, d.rem_bounces))
        tjrtn = tj.get_action(discs, len(discs), ohalf.hero.pos.x, ohalf.hero.pos.y, ohalf.hero.charge, shalf.hero.pos.x, shalf.hero.pos.y, shalf.hero.charge)
        movement = vector(tjrtn[0], tjrtn[1])
        throw_dir = vector(tjrtn[2], tjrtn[3])
        throw_discs = []
        # print(shalf.hero.pos)
        if shalf.hero.charge == 10:
            throw_discs = [throw_dir]
        return tick_action(movement, throw_discs)

class hetaro_bot(base_bot):
    def __init__(self, name, model_path, pt):
        import numpy as np
        import torch
        import gym
        from gym import spaces
        from stable_baselines3 import PPO
        class MockRicochetEnv(gym.Env):
            metadata = {'render.modes': ['human']}
            def __init__(self):
                self.action_space = spaces.Box(low = -1, high = 1, shape = (8,), dtype = np.float32)
                self.observation_space = spaces.Box(low = -1, high = 1, shape = (2, 11, 7), dtype = np.float32)
            def step(self, action):
                obs = np.zeros((2, 11, 7), dtype=np.float32)
                return obs, 0, False, {}
            def reset(self):
                obs = np.zeros((2, 11, 7), dtype=np.float32)
                return obs
            def render(self, mode='human'):
                pass
            def close(self):
                pass
        env = MockRicochetEnv()
        agent = PPO('MlpPolicy', env, policy_kwargs={'net_arch':[64]*4+[32]*8})
        if pt:
            params = torch.load(model_path + '.pt')
            agent.set_parameters(params)
        else:
            agent = PPO.load(model_path, env)
        self._name = name
        self._agent = agent
    def get_bot_name(self):
        return self._name
    def get_player_name(self):
        return 'komiamiko'
    def reset(self, no):
        pass
    def get_action(self, shalf, ohalf):
        import numpy as np
        agent = self._agent
        # define the helper functions
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
        # build the observation
        obs = _ricochet_to_agent_observation([shalf, ohalf])
        # get the action
        raw_action, _ = agent.predict(obs)
        # convert action to tournament format
        action = _ricochet_from_agent_action(raw_action, shalf.hero.charge // DISC_COST)
        #done
        return action

class sugarcane_bot(base_bot):
    """
    Sugarcane, the successor of Reid
    """
    def __init__(self, name: str, use_hotshot: bool = False, spread_multiplier: float = 0.95, noreflect: float = 3):
        self.name = name
        self.use_hotshot = use_hotshot
        self.spread_multiplier = spread_multiplier
        self.noreflect = noreflect
    def get_bot_name(self) -> str:
        return self.name
    def get_player_name(self) -> str:
        return 'komiamiko'
    def reset(self, no: bool):
        import random
        self.phase = 0
        self._random = random.Random()
    def get_action(self, shalf, ohalf) -> tick_action:
        import numpy as np
        random = self._random
        # project T steps ahead for each of D directions, pessimistically
        T = 12
        D = 32
        da = random.uniform(0, np.pi*2)
        tdiscs = [[disc.pos.x, disc.pos.y, disc.vel.x, disc.vel.y] for disc in shalf.discs]
        if not tdiscs or ohalf.hero.charge >= DISC_COST:
            dos = shalf.hero.pos - ohalf.hero.pos
            ivel = dos * (DISC_SPEED / dos.magnitude())
            tdiscs.append([ohalf.hero.pos.x, ohalf.hero.pos.y, ivel.x, ivel.y])
        tdiscs = np.array(tdiscs)
        N = tdiscs.shape[0]
        di_pos = tdiscs[:,0:2]
        di_vel = tdiscs[:,2:4]
        s_pos = np.array([shalf.hero.pos.x, shalf.hero.pos.y])
        o_pos = np.array([ohalf.hero.pos.x, ohalf.hero.pos.y])
        s_vel = np.array([[np.cos(a+da), np.sin(a+da)] for a in np.linspace(0, np.pi*2, D, False)]) * MOVE_SPEED
        dt = np.arange(T)+1
        di_proj = di_pos + di_vel * dt.reshape((T, 1, 1))
        di_proj = np.abs(di_proj)
        di_proj = np.minimum(di_proj, np.array([WIDTH, HEIGHT])*2 - di_proj)
        s_proj = s_pos + s_vel.reshape((D, 1, 1, 2)) * dt.reshape((1, T, 1, 1))
        s_proj = np.maximum(1, s_proj)
        s_proj = np.minimum(np.array([WIDTH-1,HEIGHT-1]), s_proj)
        ds_proj = di_proj - s_proj
        os_proj = o_pos - s_proj
        dsq_proj = np.sum(ds_proj ** 2, axis=-1) # (D, T, N)
        osq_proj = np.sum(os_proj ** 2, axis=-1) # (D, T, 1)
        hits = dsq_proj < INTERSECT_DISTANCE ** 2
        nhits = dsq_proj < (2 * INTERSECT_DISTANCE) ** 2
        IDEAL_DISTANCE = INTERSECT_DISTANCE * 2.1 / MOVE_SPEED * DISC_SPEED
        hunter = np.abs(osq_proj ** 0.5 - IDEAL_DISTANCE)
        loss = hits + 0.1 * nhits + 0.0001 * hunter
        loss = np.sum(loss, axis=-1)
        loss *= 0.9 ** dt
        loss = np.sum(loss, axis=-1)
        ii = np.argmin(loss)
        mvec = s_vel[ii]
        movement = vector(float(mvec[0]), float(mvec[1]))
        # do attack pattern
        throw_discs = []
        if self.phase == 0 and shalf.hero.charge >= DISC_COST * 2:
            if self.use_hotshot:
                self.phase = 1
            mypos = shalf.hero.pos
            target = ohalf.hero.pos
            delta = target - mypos
            delta_mag = delta.magnitude()
            project_at = max(INTERSECT_DISTANCE * 2, delta_mag)
            SPREAD_FACTOR = math.sqrt(1 - (MOVE_SPEED + DISC_SPEED) / (2 * INTERSECT_DISTANCE)) * self.spread_multiplier
            spread = SPREAD_FACTOR * INTERSECT_DISTANCE / project_at
            throw_discs.append(vector(delta.x + delta.y * spread, delta.y - delta.x * spread))
            throw_discs.append(vector(delta.x - delta.y * spread, delta.y + delta.x * spread))
        elif self.phase == 1 and shalf.hero.charge == CHARGE_MAX:
            self.phase = 0
            opos = ohalf.hero.pos
            dists = [opos.x,WIDTH-opos.x,opos.y,HEIGHT-opos.y,self.noreflect]
            jj = np.argmin(dists)
            if jj == 0:
                opos = vector(-opos.x, opos.y)
            elif jj == 1:
                opos = vector(WIDTH*2-opos.x, opos.y)
            elif jj == 2:
                opos = vector(opos.x, -opos.y)
            elif jj == 3:
                opos = vector(opos.x, HEIGHT*2-opos.y)
            throw_dir = opos - shalf.hero.pos
            throw_discs.append(throw_dir)
        return tick_action(movement, throw_discs)

# END --- define bots above!

def run_match(p0: base_bot, p1: base_bot) -> typing.Tuple[int, int]:
    state = match_state((p0, p1))
    while True:
        res = state.update()
        if res is not None:
            return res

def run_match_recorded(p0: base_bot, p1: base_bot, ii: int, num_matches:int, c0: int, c1: int) -> typing.Tuple[int, int]:
    state = recorded_match_state((p0, p1))
    while True:
        res = state.update()
        if res is not None:
            state.match_result = res
            state.export(ii, num_matches, c0, c1)
            return res

def make_all_bots() -> typing.List[base_bot]:
    return [
        random_bot(False, False),
        random_bot(False, True),
        random_bot(True, False),
        random_bot(True, True),
        melee_bot(),
        reid_bot('Reid', use_hunter=False, mc_expand=7, mc_iters=3, spread_multiplier=0.38),
        reid_bot('Reid 2'),
        tj_bot(),
        hetaro_bot('NoHetaro', 'hetaro/release/27_12', True),
        sugarcane_bot('Sugarcane'),
        sugarcane_bot('Caramelcane', use_hotshot=True),
        ]

def run_tournament(options: argparse.Namespace, all_bots: typing.List[base_bot]) -> typing.List[typing.Tuple[int, str, str]]:
    import random
    orandom = random.Random()
    score_counters = collections.Counter()
    match_counters = collections.Counter()
    all_pairs = list(itertools.combinations(all_bots, 2))
    allow_prints = options.nth_process == 0
    if allow_prints:print(f'{len(all_bots)} bots, {len(all_pairs)} matchups')
    for ii in range(TOURNAMENT_OUTER_ITERATIONS * options.nth_process // options.num_processes,
        TOURNAMENT_OUTER_ITERATIONS * (options.nth_process + 1) // options.num_processes):
        for p0, p1 in all_pairs:
            p0.reset(True)
            p1.reset(True)
            time_start = datetime.datetime.now()
            time_alerted = False
            c0 = c1 = 0
            for jj in range(TOURNAMENT_MATCHES):
                if orandom.randrange(2) == 0:
                    p0, p1 = p1, p0
                    c0, c1 = c1, c0
                d0, d1 = run_match(p0, p1)
                c0 += d0
                c1 += d1
                time_now = datetime.datetime.now()
                if time_now - time_start > TOURNAMENT_PATIENCE and not time_alerted:
                    time_alerted = True
                    if allow_prints:print(f'{p0} vs {p1} ... finished {jj+1} / {TOURNAMENT_MATCHES} matches in {time_now - time_start}')
            time_end = datetime.datetime.now()
            if ii == 0:
                time_suffix = '' if time_end - time_start < TOURNAMENT_PATIENCE else f' (took {time_end - time_start})'
                if allow_prints:print(f'{p0} vs {p1} - {c0}:{c1}{time_suffix}')
            score_counters[str(p0)] += c0
            score_counters[str(p1)] += c1
            match_counters[str(p0)] += TOURNAMENT_MATCHES
            match_counters[str(p1)] += TOURNAMENT_MATCHES
        orandom.shuffle(all_pairs)
        if allow_prints:print(f'big iteration {ii+1} / {TOURNAMENT_OUTER_ITERATIONS}')
    score_table = [(score_counters[str(p0)], str(p0), p0.get_player_name(), match_counters[str(p0)]) for p0 in all_bots]
    score_table.sort(reverse=True, key=(lambda tup:tup[0]/tup[3]))
    return score_table

def record_matches(all_bots: typing.List[base_bot], record_options: typing.List[str]):
    """
    Record the requested number of consecutive matches between these bots.
    """
    bots_by_name = {bot.get_bot_name():bot for bot in all_bots}
    aname, bname, num_matches = record_options
    num_matches = int(num_matches)
    p0 = bots_by_name[aname]
    p1 = bots_by_name[bname]
    p0.reset(True)
    p1.reset(True)
    c0 = c1 = 0
    for ii in range(num_matches):
        print(f'Running {p0} vs {p1} match {ii+1} of {num_matches}')
        p0.reset(False)
        p1.reset(False)
        d0, d1 = run_match_recorded(p0, p1, ii, num_matches, c0, c1)
        c0 += d0
        c1 += d1

def print_score_table(score_table: typing.List[typing.Tuple[int, str, str]]):
    print('\n'.join(f'# {tup[0]*1000000//tup[3]:>7} | {tup[1]:>20} by {tup[2]}' for tup in score_table))

def get_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the full tournament for the ricochet game.')
    parser.add_argument('-N', '--num-processes', type=int, default=1,
        help='How many processes to split into for running more matches.')
    parser.add_argument('-n', '--nth-process', type=int, default=0,
        help='Which process is this. Don\'t specify if you\'re running normally from the command line.')
    parser.add_argument('-r', '--record', nargs=3, default=None,
        help='Given player A name, player B name, and number of matches, run that many matches between A and B ' + \
             'in a special recording mode, and write the files to the visualize/replay directory. These scripts can then be pasted ' + \
             'into the Blender Python shell in the animation file to generate the full replay, ready to render. ' + \
             'Using this option prevents the regular tournament from running.')
    options = parser.parse_args()
    return options

def main():
    options = get_options()
    all_bots = make_all_bots()
    if options.record:
        record_matches(all_bots, options.record)
        return
    if options.num_processes != 1 and options.nth_process == 0:
        subs = []
        captures = []
        for i in range(1, options.num_processes):
            cap = open(f'tmp{i}.txt', 'w+')
            sub = subprocess.Popen(['python3', '-m', 'ricochet', '-N', str(options.num_processes), '-n', str(i)], stdout=cap)
            subs.append(sub)
            captures.append(cap)
    score_table = run_tournament(options, all_bots)
    if options.num_processes != 1 and options.nth_process == 0:
        name_map = {y: z for x, y, z, w in score_table}
        score_map = collections.Counter({y: x for x, y, z, w in score_table})
        match_map = collections.Counter({y: w for x, y, z, w in score_table})
        for i, sub, cap in zip(range(1, options.num_processes), subs, captures):
            print(f'Waiting on process #{i}')
            retval = sub.wait()
            cap.seek(0)
            iscore_table = json.loads(cap.read().strip())
            cap.close()
            score_map += collections.Counter({y: x for x, y, z, w in iscore_table})
            match_map += collections.Counter({y: w for x, y, z, w in iscore_table})
        score_table = [(score_map[y], y, name_map[y], match_map[y]) for y in score_map]
        score_table.sort(reverse=True, key=(lambda tup:tup[0]/tup[3]))
    if options.nth_process == 0:
        print_score_table(score_table)
    else:
        print(json.dumps(score_table))

if __name__ == '__main__':
    main()
