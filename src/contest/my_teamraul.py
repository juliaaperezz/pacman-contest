import random
import util
import heapq
from capture_agents import CaptureAgent
from game import Directions, Actions
from util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    Creates a team of agents.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def _init_(self, index, time_for_computing=.1):
        super()._init_(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        raise NotImplementedError

    def get_weights(self, game_state, action):
        raise NotImplementedError

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food and targets scared ghosts when possible.
    """

    def choose_action(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()

        # Detectar fantasmas no asustados cerca
        ghosts = [
            enemy.get_position()
            for enemy in [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            if enemy.get_position() is not None and enemy.scared_timer == 0
        ]
        is_in_danger = False
        if ghosts:
            min_ghost_dist = min([self.get_maze_distance(my_pos, ghost) for ghost in ghosts])
            is_in_danger = min_ghost_dist <= 5  # Define el umbral de peligro
            if is_in_danger:
                # Si está en peligro, prioriza volver al lado propio
                print("¡En peligro! Priorizar volver al lado propio.")
                return self.retreat_to_home(game_state, my_pos)

        # Si no hay peligro, prioriza comida o fantasmas asustados
        scared_ghosts = [
            enemy.get_position()
            for enemy in [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            if enemy.scared_timer > 0 and enemy.get_position() is not None
        ]

        if scared_ghosts:
            goal = min(scared_ghosts, key=lambda ghost: self.get_maze_distance(my_pos, ghost))
            path = a_star_search(my_pos, goal, game_state, get_successors, manhattan_heuristic)
            if path:
                return path[0]

        if food_list:
            goal = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
            path = a_star_search(my_pos, goal, game_state, get_successors, manhattan_heuristic)
            if path:
                return path[0]

        # Acción por defecto
        return ReflexCaptureAgent.choose_action(self, game_state)

    def retreat_to_home(self, game_state, my_pos):
        """
        Prioriza volver al lado propio del campo cuando está en peligro.
        """
        mid_x = game_state.data.layout.width // 2
        home_positions = [
            (x, y)
            for x in range(0, mid_x)  # Solo posiciones en el lado propio
            for y in range(game_state.data.layout.height)
            if not game_state.has_wall(x, y)
        ]

        # Encuentra la posición más cercana en el lado propio
        goal = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
        path = a_star_search(my_pos, goal, game_state, get_successors, manhattan_heuristic)
        if path:
            return path[0]

        # Si no hay camino, usa la acción por defecto
        return random.choice(game_state.get_legal_actions(self.index))
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_pos = successor.get_agent_state(self.index).get_position()

        # Puntuación basada en la cantidad de comida restante
        features['successor_score'] = -len(food_list)

        # Distancia a la comida más cercana
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Fantasmas no asustados cerca
        ghosts = [
            enemy.get_position()
            for enemy in [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            if enemy.get_position() is not None and enemy.scared_timer == 0
        ]
        if ghosts:
            min_distance = min([self.get_maze_distance(my_pos, ghost) for ghost in ghosts])
            features['distance_to_ghost'] = min_distance

        # Fantasmas asustados cerca
        scared_ghosts = [
            enemy.get_position()
            for enemy in [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            if enemy.scared_timer > 0 and enemy.get_position() is not None
        ]
        if scared_ghosts:
            min_distance = min([self.get_maze_distance(my_pos, ghost) for ghost in scared_ghosts])
            features['distance_to_scared_ghost'] = min_distance

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,  # Maximiza puntos de comida
            'distance_to_food': -1,  # Minimiza distancia a la comida
            'distance_to_ghost': 200,  # Penaliza estar cerca de fantasmas
            'distance_to_scared_ghost': -10,  # Maximiza la proximidad a fantasmas asustados
            'return_home': -500,  # Prioriza volver al campo propio si es necesario
        }

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that defends its side and avoids invaders when scared.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        scared = my_state.scared_timer > 0

        features['num_invaders'] = len(invaders)

        if invaders and not scared:
            # If not scared, move towards the closest invader
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists) if dists else 0

        if scared:
            # If scared, avoid attacking invaders but remain on defense
            features['on_defense'] = 1
            features['invader_distance'] = 0  # Neutralize this factor while scared

        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Adjusted weights for defensive behavior.
        """
        return {
            'num_invaders': -1000,  # Penalize having invaders on our side
            'on_defense': 100,  # Encourage staying on defense
            'invader_distance': -10,  # Penalize distance to invaders (only when not scared)
            'stop': -100,  # Discourage stopping
            'reverse': -2,  # Discourage reversing
        }

#######################
# A* Helper Functions #
#######################

def a_star_search(start, goal, game_state, get_successors, heuristic):
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), start, [], 0))
    closed_set = set()

    while open_list:
        _, current_pos, path, current_cost = heapq.heappop(open_list)

        if current_pos == goal:
            return path

        if current_pos in closed_set:
            continue
        closed_set.add(current_pos)

        for next_pos, action, step_cost in get_successors(game_state, current_pos):
            if next_pos not in closed_set:
                total_cost = current_cost + step_cost
                priority = total_cost + heuristic(next_pos, goal)
                heapq.heappush(open_list, (priority, next_pos, path + [action], total_cost))

    return []

def get_successors(game_state, position):
    ACTION_DELTA = {
        Directions.NORTH: (0, 1),
        Directions.SOUTH: (0, -1),
        Directions.EAST: (1, 0),
        Directions.WEST: (-1, 0),
    }
    successors = []
    x, y = position
    for action, (dx, dy) in ACTION_DELTA.items():
        next_x, next_y = int(x + dx), int(y + dy)
        if not game_state.has_wall(next_x, next_y):
            successors.append(((next_x, next_y), action, 1))
    return successors

def manhattan_heuristic(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])