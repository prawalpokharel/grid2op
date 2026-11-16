# Copyright (c) 2019-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import copy
from typing import Dict, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from grid2op.dtypes import dt_float, dt_int
from grid2op.Exceptions import OpponentError
from grid2op.Opponent.baseOpponent import BaseOpponent


class CriticalLineOpponent(BaseOpponent):
    """
    This opponent disconnects, among the attackable lines `lines_attacked`, the connected line with the
    highest structural criticality score. The score is computed on the current topology, so it follows
    the agent's topology changes and the lines already disconnected.

    Two scores are available (argument `score` of :func:`CriticalLineOpponent.init`):

    - ``"betweenness"``: edge betweenness of the line in the bus graph. Parallel lines between the same
      two buses share the betweenness of their corridor equally.
    - ``"criticality"``: ``alpha * C + beta * H + gamma * R`` (a centrality-entropy index), where

      - ``C`` is a composite centrality of the line in the line graph of the bus graph
        (0.35 PageRank + 0.30 betweenness + 0.20 degree + 0.15 closeness, divided by its maximum),
      - ``H`` is the Shannon entropy of the line's usage rate over the last `history_length` steps
        (10 bins on [0, 1], divided by ``log2(10)``),
      - ``R`` is a criticality in [0, 1] declared by the user for each line (argument `governance`,
        0 for lines not listed).

    The time of the attack follows the same rule as :class:`WeightedRandomOpponent`: when an attack
    becomes possible, its time is sampled uniformly in the next `attack_period` steps. This time is drawn
    from a random generator dedicated to timing (seeded from the opponent's own generator at each reset),
    so two opponents of this class with different scores attack at the same steps of a seeded episode.
    This makes paired comparisons of target rules possible.

    Lines whose scores are within ``1e-9`` of the best score are considered tied, and the one with the lowest
    position in `lines_attacked` is attacked, so the choice does not depend on floating point rounding.

    Examples
    --------

    .. code-block:: python

        import grid2op
        from grid2op.Action import PowerlineSetAction
        from grid2op.Opponent import CriticalLineOpponent, BaseActionBudget

        env = grid2op.make("l2rpn_case14_sandbox",
                           opponent_attack_cooldown=12 * 24,
                           opponent_attack_duration=12 * 4,
                           opponent_budget_per_ts=0.17,
                           opponent_init_budget=144.,
                           opponent_action_class=PowerlineSetAction,
                           opponent_class=CriticalLineOpponent,
                           opponent_budget_class=BaseActionBudget,
                           kwargs_opponent={"lines_attacked": ["1_3_3", "1_4_4", "3_6_15"],
                                            "score": "criticality"})

    References
    ----------
    The centrality-entropy index is described in P. Pokharel, "Evaluation of the CEI Framework: Enterprise
    Cloud Governance in Multi-Tier Distributed Environments", Future Technologies Conference (FTC), 2026.
    """

    SCORES = ("criticality", "betweenness")
    _TIE_TOL = 1e-9
    _N_BINS = 10

    def __init__(self, action_space):
        BaseOpponent.__init__(self, action_space)
        self._do_nothing = None
        self._attacks = None
        self._lines_ids = None
        self._attack_period = None
        self._next_attack_time = None
        self._timing_prng = None
        self._score = None
        self._weights = None
        self._governance = None
        self._history_length = None
        self._rho_history = None
        self._history_pos = 0
        self._history_count = 0

    def init(
        self,
        partial_env,
        lines_attacked: Sequence[str] = (),
        attack_period: int = 12 * 24,
        score: str = "criticality",
        weights: Tuple[float, float, float] = (0.40, 0.35, 0.25),
        governance: Optional[Dict[str, float]] = None,
        history_length: int = 12 * 24,
        **kwargs,
    ):
        """
        Parameters
        ----------
        partial_env:
            The environment that creates the opponent (not used).

        lines_attacked: ``list``
            The names of the lines the opponent may disconnect.

        attack_period: ``int``
            When an attack becomes possible, it happens at a step sampled uniformly in the next
            `attack_period` steps (same meaning as for :class:`WeightedRandomOpponent`).

        score: ``str``
            ``"criticality"`` (default) or ``"betweenness"``, see the class documentation.

        weights: ``tuple``
            The weights ``(alpha, beta, gamma)`` of the centrality, entropy and declared criticality terms.
            Only used when `score` is ``"criticality"``. They must be non negative and not all 0.

        governance: ``dict``
            A declared criticality in [0, 1] for some lines, by line name. Lines not listed get 0.

        history_length: ``int``
            Number of past steps used to compute the entropy of each line's usage rate.
        """
        if score not in type(self).SCORES:
            raise OpponentError(f"Unknown score \"{score}\". It should be one of {type(self).SCORES}.")
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (3,) or (weights < 0.0).any() or not (weights > 0.0).any():
            raise OpponentError("weights should be three non negative numbers (alpha, beta, gamma), "
                                "not all equal to 0.")
        if attack_period <= 0:
            raise OpponentError("Opponent attack period needs to be > 0")
        if history_length <= 0:
            raise OpponentError("history_length needs to be > 0")

        self._set_line_id(lines_attacked)
        self._prebuild_action()
        self._lines_ids = np.array(self._lines_ids, dtype=dt_int)
        self._attack_period = int(attack_period)
        self._score = score
        self._weights = weights
        self._history_length = int(history_length)

        n_line = type(self.action_space).n_line
        self._governance = np.zeros(n_line)
        for l_name, val in (governance or {}).items():
            l_id = (self.action_space.name_line == l_name).nonzero()[0]
            if not len(l_id):
                raise OpponentError(f"Unable to find the powerline named \"{l_name}\" on the grid.")
            if not 0.0 <= val <= 1.0:
                raise OpponentError(f"The declared criticality of \"{l_name}\" should be in [0, 1].")
            self._governance[l_id[0]] = val
        self._rho_history = np.zeros((self._history_length, n_line), dtype=dt_float)

    def reset(self, initial_budget):
        self._next_attack_time = None
        self._timing_prng = np.random.RandomState(self.space_prng.randint(np.iinfo(dt_int).max))
        if self._rho_history is not None:
            self._rho_history[:] = 0.0
        self._history_pos = 0
        self._history_count = 0

    def tell_attack_continues(self, observation, agent_action, env_action, budget):
        self._next_attack_time = None
        self._record(observation)

    def attack(self, observation, agent_action, env_action, budget, previous_fails):
        """
        Disconnect the connected attackable line with the highest score when the sampled attack time is
        reached (see the class documentation). The duration is the longest one allowed.
        """
        if observation is None:
            return None, 0
        self._record(observation)

        if self._next_attack_time is None:
            self._next_attack_time = 1 + self._timing_prng.randint(self._attack_period)
        self._next_attack_time -= 1
        if self._next_attack_time > 0:
            return None, 0

        status = observation.line_status[self._lines_ids]
        if not status.any():
            return None, 0
        position = self._choose_line(observation, status)
        if position is None:
            return None, 0
        return self._attacks[position], None

    def line_scores(self, observation) -> np.ndarray:
        """
        The score of every line of the grid (disconnected lines get ``-inf``). Higher means more critical.
        """
        graph = self._bus_graph(observation)
        if self._score == "betweenness":
            return self._edge_betweenness(graph, observation)
        alpha, beta, gamma = self._weights
        res = gamma * self._governance
        if alpha > 0.0:
            res = res + alpha * self._line_centrality(graph, observation)
        if beta > 0.0:
            res = res + beta * self._usage_entropy()
        res[~observation.line_status] = -np.inf
        return res

    def _choose_line(self, observation, status) -> Optional[int]:
        """Position, in `lines_attacked`, of the line to attack (or ``None``)."""
        scores = np.asarray(self.line_scores(observation), dtype=np.float64)[self._lines_ids]
        scores[~status] = -np.inf
        best = scores.max()
        if not np.isfinite(best):
            return None
        return int(np.flatnonzero(scores >= best - type(self)._TIE_TOL)[0])

    def _record(self, observation):
        if observation is None or self._rho_history is None:
            return
        self._rho_history[self._history_pos] = observation.rho
        self._history_pos = (self._history_pos + 1) % self._history_length
        self._history_count = min(self._history_count + 1, self._history_length)

    def _bus_graph(self, observation) -> nx.MultiGraph:
        """Buses as nodes and connected lines as edges, keyed by the line id."""
        cls = type(observation)
        graph = nx.MultiGraph()
        for l_id in np.flatnonzero(observation.line_status):
            bus_or = int(cls.line_or_to_subid[l_id]) + (int(observation.line_or_bus[l_id]) - 1) * cls.n_sub
            bus_ex = int(cls.line_ex_to_subid[l_id]) + (int(observation.line_ex_bus[l_id]) - 1) * cls.n_sub
            graph.add_edge(bus_or, bus_ex, key=int(l_id))
        return graph

    @staticmethod
    def _edge_betweenness(graph: nx.MultiGraph, observation) -> np.ndarray:
        res = np.full(type(observation).n_line, -np.inf)
        simple = nx.Graph(graph)
        betweenness = nx.edge_betweenness_centrality(simple, normalized=True)
        for (u, v), val in betweenness.items():
            parallel = graph[u][v]
            for l_id in parallel:
                res[l_id] = val / len(parallel)
        return res

    @staticmethod
    def _line_centrality(graph: nx.MultiGraph, observation) -> np.ndarray:
        res = np.zeros(type(observation).n_line)
        lines = nx.line_graph(graph)
        if lines.number_of_nodes() == 0:
            return res
        try:
            pagerank = nx.pagerank(lines, alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            pagerank = {n: 1.0 / lines.number_of_nodes() for n in lines}
        betweenness = nx.betweenness_centrality(lines)
        degree = nx.degree_centrality(lines)
        closeness = nx.closeness_centrality(lines)
        composite = {n: 0.35 * pagerank[n] + 0.30 * betweenness[n] + 0.20 * degree[n] + 0.15 * closeness[n]
                     for n in lines}
        top = max(composite.values())
        for (_, _, l_id), val in composite.items():
            res[l_id] = val / top if top > 0.0 else 0.0
        return res

    def _usage_entropy(self) -> np.ndarray:
        n_line = self._rho_history.shape[1]
        res = np.zeros(n_line)
        if self._history_count == 0:
            return res
        hist = np.clip(self._rho_history[:self._history_count], 0.0, 1.0)
        bins = np.minimum((hist * self._N_BINS).astype(dt_int), self._N_BINS - 1)
        for l_id in range(n_line):
            counts = np.bincount(bins[:, l_id], minlength=self._N_BINS)
            p = counts[counts > 0] / self._history_count
            res[l_id] = -(p * np.log2(p)).sum() / np.log2(self._N_BINS)
        return res

    def get_state(self):
        return (self._next_attack_time,
                None if self._timing_prng is None else self._timing_prng.get_state(),
                None if self._rho_history is None else self._rho_history.copy(),
                self._history_pos,
                self._history_count)

    def set_state(self, my_state):
        if my_state is None:
            return
        next_time, prng_state, history, pos, count = my_state
        self._next_attack_time = next_time
        if prng_state is not None:
            self._timing_prng = np.random.RandomState()
            self._timing_prng.set_state(prng_state)
        if history is not None:
            self._rho_history = history.copy()
        self._history_pos = pos
        self._history_count = count

    def _custom_deepcopy_for_copy(self, new_obj, dict_=None):
        new_obj._attacks = copy.deepcopy(self._attacks)
        new_obj._lines_ids = copy.deepcopy(self._lines_ids)
        new_obj._attack_period = self._attack_period
        new_obj._score = self._score
        new_obj._weights = copy.deepcopy(self._weights)
        new_obj._governance = copy.deepcopy(self._governance)
        new_obj._history_length = self._history_length
        super()._custom_deepcopy_for_copy(new_obj, dict_)
