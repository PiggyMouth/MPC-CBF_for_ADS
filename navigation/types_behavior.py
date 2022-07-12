# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains the different parameters sets for each behavior. """


class Cautious(object):
    """Class for Cautious agent."""
    # max_speed = 40
    # speed_lim_dist = 6
    # speed_decrease = 12
    # safety_time = 3
    # min_proximity_threshold = 12
    # braking_distance = 6
    # overtake_counter = -1
    # tailgate_counter = 0
    max_speed = 40
    speed_lim_dist = 10
    speed_decrease = 20
    safety_time = 8
    min_proximity_threshold = 15
    braking_distance = 10
    # never overtakes
    overtake_counter = -1
    tailgate_counter = 0



class Normal(object):
    """Class for Normal agent."""
    # max_speed = 50
    # speed_lim_dist = 3
    # speed_decrease = 10
    # safety_time = 3
    # min_proximity_threshold = 10
    # braking_distance = 5
    # overtake_counter = 0
    # tailgate_counter = 0
    max_speed = 50
    speed_lim_dist = 3
    speed_decrease = 10
    safety_time = 4
    min_proximity_threshold = 8
    braking_distance = 5
    overtake_counter = 0
    tailgate_counter = 0


class Aggressive(object):
    """Class for Aggressive agent."""
    # max_speed = 70
    # speed_lim_dist = 1
    # speed_decrease = 8
    # safety_time = 3
    # min_proximity_threshold = 8
    # braking_distance = 4
    # overtake_counter = 0
    # tailgate_counter = -1
    max_speed = 70
    speed_lim_dist = 1
    speed_decrease = 5
    safety_time = 2
    min_proximity_threshold = 3
    braking_distance = 3
    overtake_counter = 0
    # never tailgates
    tailgate_counter = -1
