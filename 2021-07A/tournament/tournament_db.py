"""
Database for the tournament runner.
Will be loaded from the disk database if it exists,
otherwise a fresh database will be created.
"""

import collections
import pathlib
import typing
from .glicko2 import Player

class MatchupStats(object):
    """
    Statistics for a single match up between 2 players.
    """
    def __init__(self, wdl: typing.Optional[list[int]] = None,
        reason: typing.Optional[collections.abc.Mapping[str, int]] = None):
        if wdl is None:
            wdl = [0] * 3
        self.wdl = wdl
        if reason is None:
            reason = {}
        self.reason = collections.Counter(reason)

class Database(object):
    """
    Tournament database for ShipTurret.
    """
    def __init__(self, db_dir: typing.Union[str, pathlib.Path] = '2021-07A/tournament/database',
        players: typing.Optional[dict[str, Player]] = None,
        matchups: typing.Optional[dict[tuple[str, str], MatchupStats]] = None):
        db_dir = pathlib.Path(db_dir)
        self.db_dir = db_dir
        if players is None:
            players = {}
        self.players = players
        if matchups is None:
            matchups = {}
        self.matchups = matchups
