from numpy import random as np_random
import random
import numpy as np
import copy
import string
import math
from constants import MAX_SPEED

N_INPUTS = 9
N_ACTIONS = 4  # [up, down, left, right]
N_HIDDEN = 16
STOP_PENALTY = 0.15
PEDAL_BONUS = 0.08
STEER_BONUS = 0.04
SPEED_EPS = 5.0
STOP_STREAK_GRACE = 12
STOP_STEP_PENALTY = 0.02
STOP_STREAK_PENALTY = 0.004
STOP_STREAK_POWER = 2.0
TURN_ANGLE_EPS = 0.06
CORNER_SPEED_TARGET = 0.35
CORNER_OVERSPEED_PENALTY = 1.3
NO_BRAKE_IN_CORNER_PENALTY = 0.35
THROTTLE_IN_CORNER_PENALTY = 0.25


class AIbrain_QERQ:
    global_best_score_seen = None
    global_epochs_without_improvement = 0
    global_last_improved = False

    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
        self.decider = 0
        self.x = 0
        self.y = 0
        self.speed = 0
        self.prev_pos = None
        self.prev_move = None
        self.turn_penalty = 0.0
        self.mutation_generation = 0
        self.last_mutation_rate = None
        self.last_mutation_chance = None
        self.best_score_seen = None
        self.epochs_without_improvement = 0
        self.last_improved = False
        self.recent_scores = []
        self.recent_margins = []
        self.score_margin_corr = None
        self.last_avg_margin = 0.0
        self.stop_streak = 0
        self.max_stop_streak = 0
        self.stop_steps_total = 0
        self.last_angle_delta = 0.0
        self.prev_speed = 0.0

        self.action_counts = np.zeros(N_ACTIONS, dtype=int)
        self.margin_sum = 0.0
        self.margin_count = 0
        self.input_action_sum = np.zeros((N_ACTIONS, N_INPUTS))
        self.input_action_count = np.zeros(N_ACTIONS)
        self.init_param()

    def init_param(self):
        limit_w1 = np.sqrt(6 / (N_INPUTS + N_HIDDEN))
        self.W1 = np_random.uniform(-limit_w1, limit_w1, size=(N_HIDDEN, N_INPUTS))
        self.b1 = np.zeros(N_HIDDEN, dtype=float)

        limit_w2 = np.sqrt(6 / (N_HIDDEN + N_ACTIONS))
        self.W2 = np_random.uniform(-limit_w2, limit_w2, size=(N_ACTIONS, N_HIDDEN))
        self.b2 = np.zeros(N_ACTIONS, dtype=float)

        self.NAME = "QERQ_neuron"
        self.store()

    def _trim_history(self, values, max_len):
        if len(values) > max_len:
            del values[:len(values) - max_len]

    def _is_significant_improvement(self, current_score, best_score):
        if best_score is None:
            return True
        epsilon = max(1e-4, abs(best_score) * 1e-6)
        return current_score > best_score + epsilon

    def _update_progress_stats(self, decision_margin):
        current_score = float(self.score)
        if self._is_significant_improvement(current_score, self.best_score_seen):
            self.best_score_seen = current_score
            self.epochs_without_improvement = 0
            self.last_improved = True
        else:
            self.epochs_without_improvement += 1
            self.last_improved = False

        cls = self.__class__
        if self._is_significant_improvement(current_score, cls.global_best_score_seen):
            cls.global_best_score_seen = current_score
            cls.global_epochs_without_improvement = 0
            cls.global_last_improved = True
        else:
            cls.global_epochs_without_improvement += 1
            cls.global_last_improved = False

        self.recent_scores.append(current_score)
        self.recent_margins.append(float(decision_margin))
        self._trim_history(self.recent_scores, 30)
        self._trim_history(self.recent_margins, 30)

        if len(self.recent_scores) >= 3 and len(self.recent_margins) >= 3:
            score_std = float(np.std(self.recent_scores))
            margin_std = float(np.std(self.recent_margins))
            if score_std > 1e-6 and margin_std > 1e-6:
                corr = np.corrcoef(self.recent_scores, self.recent_margins)[0, 1]
                self.score_margin_corr = float(corr)
            else:
                self.score_margin_corr = None
        else:
            self.score_margin_corr = None

    def _is_performance_volatile(self):
        if len(self.recent_scores) < 5:
            return False
        mean_score = float(np.mean(self.recent_scores))
        std_score = float(np.std(self.recent_scores))
        scale = max(abs(mean_score), 1.0)
        return (std_score / scale) > 0.35

    def _sigmoid(self, x):
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))


    def _compute_output_temperature(self):
        temp = 1.0
        plateau_threshold = 12
        global_ewo = int(getattr(self.__class__, "global_epochs_without_improvement", 0) or 0)
        local_ewo = int(getattr(self, "epochs_without_improvement", 0) or 0)
        ewo = max(global_ewo, local_ewo)

        if ewo >= plateau_threshold:
            temp *= 1.0 + min(0.9, 0.05 * (ewo - plateau_threshold + 1))

        if getattr(self, "last_improved", False) or getattr(self.__class__, "global_last_improved", False):
            temp *= 0.90

        corr = getattr(self, "score_margin_corr", None)
        if corr is not None and corr < -0.2 and getattr(self, "last_avg_margin", 0.0) > 0.12:
            temp *= 1.0 + min(0.35, (-corr - 0.2) * 0.25)

        return float(np.clip(temp, 0.60, 2.00))

    def _apply_pairwise_constraints(self, z):
        z = np.asarray(z, dtype=float).copy()
        if z.size >= 2 and z[0] > 0.5 and z[1] > 0.5:
            if z[0] >= z[1]:
                z[1] = 0.0
            else:
                z[0] = 0.0
        if z.size >= 4 and z[2] > 0.5 and z[3] > 0.5:
            if z[2] >= z[3]:
                z[3] = 0.0
            else:
                z[2] = 0.0
        return z

    def store(self):
        self.parameters = copy.deepcopy({
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "NAME": self.NAME,
        })

    def decide(self, data):
        self.decider += 1
        x = np.asarray(data, dtype=float).ravel()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        n_in = self.W1.shape[1]
        if x.size < n_in:
            x = np.pad(x, (0, n_in - x.size), 'constant')
        elif x.size > n_in:
            x = x[:n_in]

        hidden_raw = self.W1 @ x + self.b1
        hidden_activated = np.tanh(hidden_raw)
        z_raw = self.W2 @ hidden_activated + self.b2
        temperature = self._compute_output_temperature()
        z = self._sigmoid(z_raw / temperature)

        z = self._apply_pairwise_constraints(z)
        if z.size >= 2:
            top2 = np.partition(z, -2)[-2:]
            margin = float(np.max(top2) - np.min(top2))
        elif z.size == 1:
            margin = float(abs(z[0]))
        else:
            margin = 0.0

        self.margin_sum += margin
        self.margin_count += 1

        if z.size >= N_ACTIONS:
            active = (z > 0.5).astype(int)
            for a in range(N_ACTIONS):
                if active[a]:
                    self.action_counts[a] += 1
                    self.input_action_sum[a] += x
                    self.input_action_count[a] += 1

        self.last_actions = (z > 0.5).astype(int)
        return z

    def mutate(self):
        self.mutation_generation += 1
        base_mutation_rate = 0.25
        base_mutation_chance = 0.15
        decay = 1.0 / (1.0 + 0.005 * self.mutation_generation)
        mutation_rate = max(0.05, base_mutation_rate * decay)
        mutation_chance = max(0.04, base_mutation_chance * decay)

        multiplier = 1.0
        if self.last_improved:
            multiplier *= 0.85

        if self.__class__.global_last_improved:
            multiplier *= 0.9

        plateau_threshold = 12
        epochs_without_improvement = max(
            self.epochs_without_improvement,
            self.__class__.global_epochs_without_improvement,
        )
        if epochs_without_improvement >= plateau_threshold:
            plateau_boost = 1.0 + 0.05 * (
                epochs_without_improvement - plateau_threshold + 1
            )
            multiplier *= min(plateau_boost, 1.6)

        if self._is_performance_volatile():
            multiplier *= 0.8

        mutation_rate = max(0.03, mutation_rate * multiplier)
        mutation_chance = max(0.03, mutation_chance * multiplier)
        self.last_mutation_rate = mutation_rate
        self.last_mutation_chance = mutation_chance

        w1_mask = np_random.rand(*self.W1.shape) < mutation_chance
        b1_mask = np_random.rand(*self.b1.shape) < mutation_chance
        self.W1 += w1_mask * np_random.normal(0, mutation_rate, size=self.W1.shape)
        self.b1 += b1_mask * np_random.normal(0, mutation_rate, size=self.b1.shape)

        w2_mask = np_random.rand(*self.W2.shape) < mutation_chance
        b2_mask = np_random.rand(*self.b2.shape) < mutation_chance
        self.W2 += w2_mask * np_random.normal(0, mutation_rate, size=self.W2.shape)
        self.b2 += b2_mask * np_random.normal(0, mutation_rate, size=self.b2.shape)

        self.NAME += "_MUT_" + ''.join(random.choices(self.chars, k=2))
        self.store()

    def calculate_score(self, distance, time, no):
        safe_time = max(time, 1e-6)
        progress = max(distance, 0.0)
        safe_distance = max(progress, 1e-6)
        speed_ratio = min(abs(self.speed) / MAX_SPEED, 1.0)
        if progress <= 0.0:
            crash_penalty = safe_time
        else:
            crash_penalty = 1.0 / safe_distance
        time_penalty = safe_time / (safe_distance ** 0.8)
        speed_threshold = MAX_SPEED * 0.7
        speed_penalty = 0.0
        if self.speed > speed_threshold:
            speed_excess_ratio = (self.speed - speed_threshold) / speed_threshold
            speed_penalty = crash_penalty * (speed_excess_ratio + speed_excess_ratio ** 2)
        turn_penalty = (self.turn_penalty / safe_distance) * (1.5 + 2.5 * speed_ratio ** 2)
        progress_bonus = math.sqrt(progress) * 0.1
        self.score = progress + progress_bonus - crash_penalty - time_penalty - speed_penalty - turn_penalty + no
        if abs(self.speed) < SPEED_EPS:
            self.score -= STOP_PENALTY * safe_time

        trying = 0
        throttle = False
        braking = False
        if hasattr(self, "last_actions"):
            throttle = bool(self.last_actions[0])
            braking = bool(self.last_actions[1])
            trying = 1 if (throttle or braking) else 0

        trying_mult = 1.0 - 0.4 * trying

        self.score -= STOP_STEP_PENALTY * trying_mult * float(getattr(self, "stop_steps_total", 0))
        max_streak = int(getattr(self, "max_stop_streak", 0))
        excess = max(0, max_streak - STOP_STREAK_GRACE)
        if excess > 0:
            self.score -= STOP_STREAK_PENALTY * trying_mult * (excess ** STOP_STREAK_POWER)

        moving = abs(self.speed) >= SPEED_EPS

        if hasattr(self, "last_actions"):
            if moving and (throttle or braking):
                self.score += PEDAL_BONUS * speed_ratio
            if moving and (self.last_actions[2] or self.last_actions[3]):
                self.score += STEER_BONUS * speed_ratio

        curv = float(getattr(self, "last_angle_delta", 0.0))
        in_corner = curv > TURN_ANGLE_EPS

        if in_corner:
            overspeed = max(0.0, speed_ratio - CORNER_SPEED_TARGET)
            corner_weight = min(1.0, curv / 0.35)
            self.score -= CORNER_OVERSPEED_PENALTY * corner_weight * (overspeed ** 2) * safe_time

            if hasattr(self, "last_actions"):
                braking = bool(self.last_actions[1])
                throttle = bool(self.last_actions[0])
                if not braking and speed_ratio > CORNER_SPEED_TARGET:
                    self.score -= NO_BRAKE_IN_CORNER_PENALTY * corner_weight * safe_time
                if throttle and speed_ratio > CORNER_SPEED_TARGET:
                    self.score -= THROTTLE_IN_CORNER_PENALTY * corner_weight * safe_time
        if abs(getattr(self, "prev_speed", 0.0)) < SPEED_EPS and abs(self.speed) >= SPEED_EPS:
            self.score += 0.15

    def passcardata(self, x, y, speed):
        self.last_angle_delta = 0.0
        if self.prev_pos is not None:
            dx = x - self.prev_pos[0]
            dy = y - self.prev_pos[1]
            move_norm = math.hypot(dx, dy)
            if move_norm > 1e-6:
                if self.prev_move is not None:
                    prev_dx, prev_dy = self.prev_move
                    prev_norm = math.hypot(prev_dx, prev_dy)
                    if prev_norm > 1e-6:
                        dot = (dx * prev_dx) + (dy * prev_dy)
                        cos_angle = dot / (move_norm * prev_norm)
                        cos_angle = max(-1.0, min(1.0, cos_angle))
                        angle_delta = math.acos(cos_angle)
                        self.last_angle_delta = angle_delta
                        speed_ratio = min(abs(speed) / MAX_SPEED, 1.0)
                        self.turn_penalty += angle_delta * (speed_ratio ** 2)
                self.prev_move = (dx, dy)
        self.prev_pos = (x, y)
        self.x = x
        self.y = y
        old_speed = self.speed
        self.speed = speed
        self.prev_speed = old_speed
        if abs(speed) < SPEED_EPS:
            self.stop_streak += 1
            self.stop_steps_total += 1
            if self.stop_streak > self.max_stop_streak:
                self.max_stop_streak = self.stop_streak
        else:
            self.stop_streak = 0

    def get_diagnostics(self):
        avg_margin = (
            self.margin_sum / self.margin_count
            if self.margin_count > 0 else 0.0
        )
        self._update_progress_stats(avg_margin)
        self.last_avg_margin = avg_margin
        total = np.sum(self.action_counts)
        action_dist = (
            self.action_counts / total
            if total > 0 else np.zeros(N_ACTIONS)
        )
        if self.last_mutation_rate is not None:
            print("mutation_rate:", self.last_mutation_rate)
        if self.last_mutation_chance is not None:
            print("mutation_chance:", self.last_mutation_chance)
        if self.score_margin_corr is not None:
            print("score_margin_corr:", self.score_margin_corr)
        if self.__class__.global_epochs_without_improvement is not None:
            print("epochs_without_improvement:", self.__class__.global_epochs_without_improvement)

        return {
            "decision_margin": avg_margin,
            "actions": action_dist,
            "mutation_rate": self.last_mutation_rate,
            "mutation_chance": self.last_mutation_chance,
            "score_margin_corr": self.score_margin_corr,
            "epochs_without_improvement": self.__class__.global_epochs_without_improvement,
        }

    def get_input_action_stats(self):
        stats = {}
        for a in range(N_ACTIONS):
            if self.input_action_count[a] > 0:
                stats[a] = self.input_action_sum[a] / self.input_action_count[a]
        return stats

    def reset_diagnostics(self):
        self.action_counts[:] = 0
        self.margin_sum = 0.0
        self.margin_count = 0
        self.input_action_sum[:] = 0.0
        self.input_action_count[:] = 0

    def getscore(self):
        return self.score

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params_dict = {key: parameters[key] for key in parameters.files}
        else:
            params_dict = copy.deepcopy(parameters)

        self.parameters = params_dict

        # Načtení všech matic
        self.W1 = np.array(self.parameters["W1"], dtype=float)
        self.b1 = np.array(self.parameters["b1"], dtype=float)
        self.W2 = np.array(self.parameters["W2"], dtype=float)
        self.b2 = np.array(self.parameters["b2"], dtype=float)
        self.NAME = str(self.parameters["NAME"])
