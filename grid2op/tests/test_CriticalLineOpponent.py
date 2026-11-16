# Copyright (c) 2019-2026, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import unittest
import warnings

import networkx as nx
import numpy as np

import grid2op
from grid2op.Action import PowerlineSetAction
from grid2op.Exceptions import OpponentError
from grid2op.Opponent import BaseActionBudget, CriticalLineOpponent

ATTACK_PERIOD = 12
ATTACK_DURATION = 6


def _make_env(name, kwargs_opponent):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return grid2op.make("l2rpn_case14_sandbox",
                            test=True,
                            _add_to_name=name,
                            opponent_attack_cooldown=ATTACK_PERIOD,
                            opponent_attack_duration=ATTACK_DURATION,
                            opponent_budget_per_ts=1.0,
                            opponent_init_budget=100.,
                            opponent_action_class=PowerlineSetAction,
                            opponent_class=CriticalLineOpponent,
                            opponent_budget_class=BaseActionBudget,
                            kwargs_opponent=kwargs_opponent)


def _bus_graph(obs):
    cls = type(obs)
    graph = nx.MultiGraph()
    for l_id in np.flatnonzero(obs.line_status):
        bus_or = cls.line_or_to_subid[l_id] + (obs.line_or_bus[l_id] - 1) * cls.n_sub
        bus_ex = cls.line_ex_to_subid[l_id] + (obs.line_ex_bus[l_id] - 1) * cls.n_sub
        graph.add_edge(int(bus_or), int(bus_ex), key=int(l_id))
    return graph


def _attacked_lines(env, nb_step, seed):
    """Step at most `nb_step` times doing nothing.

    Returns the (step, line id) of every new attack and the number of steps played."""
    env.seed(seed)
    env.reset()
    res = []
    step = 0
    for step in range(nb_step):
        _, _, done, info = env.step(env.action_space())
        line = info["opponent_attack_line"]
        if line is not None and info["opponent_attack_duration"] == ATTACK_DURATION:
            res.append((step, int(np.flatnonzero(line)[0])))
        if done:
            break
    return res, step


class TestCriticalLineOpponentInit(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _make_env(type(self).__name__, {"lines_attacked": ["0_1_0"]})
        self.lines = list(self.env.name_line[:6])

    def tearDown(self) -> None:
        self.env.close()

    def test_bad_arguments(self):
        opp = CriticalLineOpponent(self.env.action_space)
        with self.assertRaises(OpponentError):
            opp.init(self.env, lines_attacked=self.lines, score="unknown")
        with self.assertRaises(OpponentError):
            opp.init(self.env, lines_attacked=self.lines, weights=(0.0, 0.0, 0.0))
        with self.assertRaises(OpponentError):
            opp.init(self.env, lines_attacked=self.lines, weights=(-1.0, 0.5, 0.5))
        with self.assertRaises(OpponentError):
            opp.init(self.env, lines_attacked=self.lines, attack_period=0)
        with self.assertRaises(OpponentError):
            opp.init(self.env, lines_attacked=self.lines, governance={"not_a_line": 1.0})
        with self.assertRaises(OpponentError):
            opp.init(self.env, lines_attacked=self.lines, governance={self.lines[0]: 2.0})


class TestCriticalLineOpponentScores(unittest.TestCase):
    def setUp(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.env = grid2op.make("l2rpn_case14_sandbox", test=True, _add_to_name=type(self).__name__)
        self.obs = self.env.reset(seed=0, options={"time serie id": 0})
        self.lines = list(self.env.name_line)

    def tearDown(self) -> None:
        self.env.close()

    def _opponent(self, **kwargs):
        opp = CriticalLineOpponent(self.env.action_space)
        opp.init(self.env, lines_attacked=self.lines, **kwargs)
        opp.seed(0)
        opp.reset(0.)
        return opp

    def test_betweenness_matches_networkx(self):
        opp = self._opponent(score="betweenness")
        scores = opp.line_scores(self.obs)
        graph = _bus_graph(self.obs)
        ref = nx.edge_betweenness_centrality(nx.Graph(graph), normalized=True)
        for (u, v), val in ref.items():
            for l_id in graph[u][v]:
                self.assertAlmostEqual(scores[l_id], val / len(graph[u][v]), places=12)

    def test_centrality_matches_reference(self):
        opp = self._opponent(weights=(1.0, 0.0, 0.0))
        scores = opp.line_scores(self.obs)
        lines = nx.line_graph(_bus_graph(self.obs))
        pr = nx.pagerank(lines, alpha=0.85)
        bw = nx.betweenness_centrality(lines)
        dg = nx.degree_centrality(lines)
        cl = nx.closeness_centrality(lines)
        comp = {n: 0.35 * pr[n] + 0.30 * bw[n] + 0.20 * dg[n] + 0.15 * cl[n] for n in lines}
        top = max(comp.values())
        for (_, _, l_id), val in comp.items():
            self.assertAlmostEqual(scores[l_id], val / top, places=12)
        self.assertAlmostEqual(scores.max(), 1.0, places=12)

    def test_entropy_matches_histogram(self):
        opp = self._opponent(weights=(0.0, 1.0, 0.0), history_length=20)
        obs = self.obs
        for _ in range(25):
            opp.tell_attack_continues(obs, None, None, 0.)
            obs, *_ = self.env.step(self.env.action_space())
        scores = opp.line_scores(obs)
        history = np.clip(opp._rho_history, 0.0, 1.0)
        for l_id in range(self.env.n_line):
            hist, _ = np.histogram(history[:, l_id], bins=10, range=(0.0, 1.0))
            p = hist[hist > 0] / hist.sum()
            self.assertAlmostEqual(scores[l_id], -(p * np.log2(p)).sum() / np.log2(10), places=6)

    def test_governance_decides_when_alone(self):
        target = self.lines[5]
        opp = self._opponent(weights=(0.0, 0.0, 1.0), governance={target: 1.0})
        position = opp._choose_line(self.obs, self.obs.line_status)
        self.assertEqual(self.lines[position], target)

    def test_disconnected_lines_are_never_chosen(self):
        opp = self._opponent(score="betweenness")
        scores = opp.line_scores(self.obs)
        graph = _bus_graph(self.obs)
        bridges = {frozenset(edge) for edge in nx.bridges(nx.Graph(graph))}
        for u, v, l_id in graph.edges(keys=True):
            if frozenset((u, v)) in bridges and len(graph[u][v]) == 1:
                scores[l_id] = -np.inf  # disconnecting it would end the episode
        best = int(np.argmax(scores))
        obs, *_ = self.env.step(self.env.action_space({"set_line_status": [(best, -1)]}))
        self.assertFalse(obs.line_status[best])
        position = opp._choose_line(obs, obs.line_status)
        self.assertNotEqual(position, best)
        self.assertTrue(np.isneginf(opp.line_scores(obs)[best]))

    def test_ties_go_to_the_first_listed_line(self):
        opp = self._opponent(weights=(0.0, 0.0, 1.0))  # every score is 0
        self.assertEqual(opp._choose_line(self.obs, self.obs.line_status), 0)


class TestCriticalLineOpponentEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.lines = ["0_1_0", "0_4_1", "1_2_2", "1_3_3", "1_4_4", "3_4_6", "5_10_7", "8_9_10"]

    def _run(self, score, seed=1, nb_step=120):
        env = _make_env(f"{type(self).__name__}_{score}",
                        {"lines_attacked": self.lines, "attack_period": ATTACK_PERIOD, "score": score})
        try:
            return _attacked_lines(env, nb_step, seed)
        finally:
            env.close()

    def test_attacks_happen(self):
        attacks, _ = self._run("betweenness")
        self.assertGreater(len(attacks), 0)

    def test_same_seed_same_attacks(self):
        self.assertEqual(self._run("criticality"), self._run("criticality"))

    def test_attack_times_do_not_depend_on_the_score(self):
        crit, crit_steps = self._run("criticality")
        betw, betw_steps = self._run("betweenness")
        horizon = min(crit_steps, betw_steps)
        self.assertGreater(len(crit), 0)
        self.assertEqual([step for step, _ in crit if step <= horizon],
                         [step for step, _ in betw if step <= horizon])

    def test_copy_keeps_the_opponent_state(self):
        env = _make_env(f"{type(self).__name__}_copy",
                        {"lines_attacked": self.lines, "attack_period": ATTACK_PERIOD})
        try:
            env.seed(3)
            env.reset()
            for _ in range(5):
                env.step(env.action_space())
            env_cpy = env.copy()
            res, res_cpy = [], []
            for _ in range(60):
                _, _, done, info = env.step(env.action_space())
                _, _, done_cpy, info_cpy = env_cpy.step(env_cpy.action_space())
                res.append(info["opponent_attack_line"] is not None)
                res_cpy.append(info_cpy["opponent_attack_line"] is not None)
                self.assertEqual(done, done_cpy)
                if done:
                    break
            self.assertEqual(res, res_cpy)
            env_cpy.close()
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
