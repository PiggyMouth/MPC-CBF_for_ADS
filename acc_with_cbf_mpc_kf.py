#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard.

# pylint: disable=protected-access

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function
# from navigation.behavior_agent import BehaviorAgent
# from navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
# from navigation.basic_agent import BasicAgent
from navigation.simple_agent import SimpleAgent

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from collections import deque
import csv
import carla
import time

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import re
import time
import weakref

import glob
import os
import random
import sys

try:
    import pygame
    from pygame_recorder import ScreenRecorder
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q

except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================
import matplotlib.pyplot as plt
from casadi import *
from PIL import Image
import queue
from mpc import MPC
import pickle
from filterpy.kalman import KalmanFilter


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name


def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm


def get_distance(frame, network_model):
    # Compute the distance from the perception LEC
    with torch.no_grad():
        # Scale the obtained distances
        dist = network_model(frame).squeeze().item() * 120
    return dist


def normalizer(value):
    return math.sqrt(value.x ** 2 + value.y ** 2 + value.z ** 2)


def write2csv(speed_, time_stamp_):
    speed_data = [[time_stamp_, speed_]]
    with open('speed_data.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for row in speed_data:
            writer.writerow(row)


def Loadmodel():
    model = PerceptionNet.PerceptionNet()
    modelPath = 'C:/Users/Admin/Simplepath/IL2232/Perception_Carla-master/Trained_Model/11_27_test/chkpt_67.pt'
    state_dict = torch.load(modelPath)
    model.load_state_dict(state_dict['model'])
    return model


def state_velocity(xt, dt, T, show=0, save=1):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t, xt[0, :], linewidth=3, color='magenta')
    plt.plot(t, 20 * np.ones(t.shape[0]), 'k--')
    # plt.title('Velocity of following vehicle')
    plt.ylabel('v_f')
    plt.xlabel('t')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    axes = plt.gca()
    axes.xaxis.label.set_size(25)
    axes.yaxis.label.set_size(25)
    if show == 1:
        plt.show()
    if save == 1:
        plt.savefig('_out/velocity.png', format='png',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def state_velocity_leading(xt, dt, T, show=0, save=1):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t, xt[1, :], linewidth=3, color='orange')
    plt.plot(t, 14 * np.ones(t.shape[0]), 'k--')
    # plt.title('Velocity of leading vehicle')
    plt.ylabel('v_l')
    plt.xlabel('t')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    axes = plt.gca()
    axes.xaxis.label.set_size(25)
    axes.yaxis.label.set_size(25)
    if show == 1:
        plt.show()
    if save == 1:
        plt.savefig('_out/velocity_l.png', format='png',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def state_relative_distance(xt, dt, T, kf_x, true_x, show=0, save=1):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    # plt.plot(t, xt[2, :], linewidth=3, color='black',
    #          label='Estimated Distance')
    plt.plot(t, kf_x[2, :], linestyle='dashed', label='Noisy Distance')
    plt.plot(t, xt[2, :], linestyle='dashed',
             color='orange', label='Estimated Distance')
    plt.plot(t, true_x[2, :], linewidth=1, color='red', label='True Distance')
    plt.legend(prop={'size': 25})
    #plt.ylim(0, 150)
    # plt.title('Relative distance')
    plt.ylabel('z')
    plt.xlabel('t')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    axes = plt.gca()
    axes.xaxis.label.set_size(25)
    axes.yaxis.label.set_size(25)
    if show == 1:
        plt.show()
    if save == 1:
        plt.savefig('_out/relative_distance.png',
                    format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def slack(slack, dt, T, show=0, save=1):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t[:-1], slack[0, :-1], linewidth=3, color='orange')
    plt.title('Slack')
    plt.ylabel('B(x)')
    if show == 1:
        plt.show()
    if save == 1:
        plt.savefig('_out/slack.png', format='png',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def cbf(Bt, dt, T, show=0, save=1):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t[:-1], Bt[0, :-1], linewidth=3, color='red')
    plt.title('cbf')
    plt.ylabel('B(x)')
    if show == 1:
        plt.show()
    if save == 1:
        plt.savefig('_out/cbf.png', format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def clf(Vt, dt, T, show=0, save=1):
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t[:-1], Vt[0, :-1], linewidth=3, color='cyan')
    plt.title('clf')
    plt.ylabel('V(x)')
    if show == 1:
        plt.show()
    if save == 1:
        plt.savefig('_out/clf.png', format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def state_control(u, dt, T, show=0, save=1):
    u_max = 0.3*9.81
    t = np.arange(0, T, dt)
    fig = plt.figure(figsize=[16, 9])
    plt.grid()
    plt.plot(t[:-1], u[0, :-1], linewidth=3, color='dodgerblue')
    plt.plot(t, u_max * np.ones(t.shape[0]), 'k--')
    plt.plot(t, -1*9.81 * np.ones(t.shape[0]), 'k--')
    # plt.title('control')
    plt.ylabel('u')
    plt.xlabel('t')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    axes = plt.gca()
    axes.xaxis.label.set_size(25)
    axes.yaxis.label.set_size(25)
    if show == 1:
        plt.show()
    if save == 1:
        plt.savefig('_out/control.png', format='png',
                    dpi=300, bbox_inches='tight')

    plt.close(fig)


def kalman_filter(xt):
    dt = 0.2
    kf = KalmanFilter(dim_x=3, dim_z=1, dim_u=1)

    kf.F = np.matrix([
        [1, 0, 0],
        [0, 1, 0],
        [-dt, dt, 1]])
    kf.B = np.matrix([
        [dt],
        [0],
        [1/2 * dt ** 2]])

    kf.P = np.eye(3)

    kf.H = np.array([[0, 0, 1]])
    kf.R = 25
    kf.x = xt

    return kf


class World(object):
    def __init__(self, carla_world, hud):
        self.world = carla_world
        self.map = carla_world.get_map()
        self.mapname = carla_world.get_map().name
        self.hud = hud
        self.world.on_tick(hud.on_world_tick)
        self.world.wait_for_tick(10.0)
        self.player = None
        while self.player is None:
            print("Scenario not yet ready")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == "hero":
                    self.player = vehicle
        self.vehicle_name = self.player.type_id
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.set_sensor(
            0, notify=False)  # Change sensor type
        self.controller = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        # self.predicted_dist = self.camera_manager._prediciton

    def restart(self):
        cam_index = self.camera_manager._index
        cam_pos_index = self.camera_manager._transform_index
        start_pose = self.player.get_transform()
        start_pose.location.z += 2.0
        start_pose.rotation.roll = 0.0
        start_pose.rotation.pitch = 0.0
        # blueprint = self._get_random_blueprint()
        blueprint = self.world.get_blueprint_library().find("vehicle.audi.etron")

        self.destroy()
        self.player = self.world.spawn_actor(blueprint, start_pose)
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        # count = 0
        # if len(self.world.get_actors().filter(self.vehicle_name)) < 1:
        #     print("Scenario ended -- Terminating")
        #     time.sleep(1)
        #     count += 1
        #     if count > 3:
        #         return False
        if len(self.world.get_actors().filter(self.vehicle_name)) < 1:
            print("Scenario ended -- Terminating")
            return False

        self.hud.tick(self, self.mapname, clock)
        return True
        # if len(self.world.get_actors().filter(self.vehicle_name)) < 1:
        #     print("Scenario ended -- Terminating")
        #     return False

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @ staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, mapname, clock):
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200]
                     for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        ################################################################
        speed_ = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        time_stamp_ = datetime.timedelta(seconds=float(self.simulation_time))
        #write2csv(speed_, time_stamp_)
        ######################

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % mapname,
            'Simulation time: % 12s' % datetime.timedelta(
                seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y)),
            'Height:  % 18.0f m' % t.location.z,
            '',
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake),
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)
        ]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            def distance(l): return math.sqrt((l.x - t.location.x) **
                                              2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x)
                        for x in vehicles if x.id != world.player.id]
            for d, vehicle in vehicles:  # sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        self._notifications.tick(world, clock)

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 *
                    self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @ staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self._hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @ staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        text = ['%r' % str(x).split()[-1]
                for x in set(event.crossed_lane_markings)]
        # self._hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        # self._camera_transforms = [
        #     carla.Transform(carla.Location(x=1.6, z=1.7)),
        #     carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))]
        self._camera_transforms = [
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            carla.Transform(carla.Location(x=0, y=0.0, z=1.7),  carla.Rotation(yaw=0.0))]

        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(bp)
        self._index = None
        self.image_queue = queue.Queue()
        self._prediction = None

    def toggle_camera(self):
        self._transform_index = (
            self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(
            self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' %
                               ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image.save_to_disk('_out/%08d' % image.frame_number)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    # pygame.camera.init()
    # pygame.time.Clock().get_fps()
    world = None

    T = 11  # 52 with default speed # clash at 106s
    dt = 0.01
    size = T/dt
    x0 = np.array([[0, 0, 100]]).T
    time_steps = int(np.ceil(T / dt))

    # initialize the input matrices
    xt = np.zeros((3, time_steps))

    kf_x = np.zeros((3, time_steps))
    kf_x[:, 0] = np.copy(x0.T[0])

    true_x = np.zeros((3, time_steps))
    true_x[:, 0] = np.copy(x0.T[0])

    ut = np.zeros((1, time_steps))
    xt[:, 0] = np.copy(x0.T[0])
    i = 0

    distance_between_vehicles = 500
    u = 0
    ############# MPC PARAMETERS ####################
    Q = np.eye(3)
    R = 2*np.eye(1)
    u_min = np.array([[-1*9.81]]).T
    u_max = np.array([[0.3*9.81]]).T
    x_l = np.array([[0, 0, 5]]).T
    x_u = np.array([[20, 15, np.inf]]).T
    N = 8
    #################################################
    kf = kalman_filter(xt[:, i].reshape(3, 1))
    # ctl = MPC(Q=Q, R=R, P=Q, N=N,
    #           ulb=u_min, uub=u_max,
    #           xlb=x_l, xub=x_u)
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud)
        # -------------------For screen capturing
        # pdb.set_trace()
        # cam = pygame.camera.Camera(0, (args.width, args.height))
        # cam.start()
        # ----------------------------------------------

        controller = KeyboardControl(world)

        # agent = BehaviorAgent(world.player, behavior=args.behavior, ignore_traffic_light=True)
        agent = SimpleAgent(world.player)

        # modify vehicle properties
        physics_control = world.player.get_physics_control()
        physics_control.mass = 1650
        # leading_agent_scenario = FollowLeadingVehicle()

        spawn_points = world.map.get_spawn_points()
        # with open('spawn_points.txt', 'w') as myfile:
        #     for i in spawn_points:
        #         myfile.write(str(i)+'\n')

        random.shuffle(spawn_points)

        # SET DESTINATION HERE
       # destination = world.map.get_waypoint(carla.Location(200, -250, 3)).transform.location

        # if spawn_points[0].location != agent.vehicle.get_location():
        #     destination = spawn_points[0].location
        # else:
        #     destination = spawn_points[1].location

        # agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

        clock = pygame.time.Clock()
        recorder = ScreenRecorder(1280, 720, 60)

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events():
                return

            if not world.world.wait_for_tick(10.0):
                continue

            # agent.update_information(world)

            if (not world.tick(clock)):
                print("Failure: scenario terminated")
                break

            world.render(display)

            pygame.display.flip()

            # ----------------save image

            recorder.capture_frame(display)

            # ----------------------Current time step-------------------------------
            ctl = MPC(Q=Q, R=R, P=Q, N=N,
                      ulb=u_min, uub=u_max,
                      xlb=x_l, xub=x_u)
            control, target_vehicle = agent.run_step()

            # real acceleartion from the ego vehicle for KF
            # a_ref = world.player.get_acceleration()
            # a_ref = normalizer(a_ref)

            ################ Noise ########################
            noise = np.random.normal(0, 0.5, 1)[0]
            #distance_between_vehicles_noise = distance_between_vehicles + noise

            simulation_time = int(hud.simulation_time)
            estimated_noise_model = pickle.load(open('differences.save', 'rb'))
            estimated_noise = estimated_noise_model.predict(
                np.array(simulation_time).reshape(-1, 1))[0]
            ####################################################

            ########################DISTANCE######################
            if distance_between_vehicles >= 500:
                # Modify distance for initial situation when the leading vehicle is not spawned
                # which causes the distance between vehicles to be larger than 500.
                distance_between_vehicles = xt[:, 0][2]
            else:
                xt[:, i][0] = ego_vel_transform
                xt[:, i][1] = target_vehicle_vel

                true_x[:, i][0] = ego_vel_transform
                true_x[:, i][1] = target_vehicle_vel
            distance_between_vehicles_noise = distance_between_vehicles + \
                estimated_noise*noise
            true_x[:, i][2] = distance_between_vehicles

            kf.predict(u)

            xt_estimate = kf.x

            kf.update(distance_between_vehicles_noise)
            xt[:, i][2] = xt_estimate[2]

            kf_x[:, i][2] = distance_between_vehicles_noise

            u, status = ctl.mpc_controller(xt[:, i].reshape(3, 1))

            if status == "Infeasible_Problem_Detected":
                # infeasible
                print('infeasible')
                print(xt[:, i])
                break
            else:
                ut[:, i] = np.copy(u)
                a = float(u)
                a_min = -1*9.81

                if a > 0:
                    control.throttle = a/10 + 0.5
                    control.brake = 0
                elif -0.3*9.81 > a >= a_min:
                    control.throttle = 0
                    control.brake = 1
                else:
                    # pdb.set_trace()
                    control.throttle = 0
                    control.brake = a/-10

            world.player.apply_control(control)

            # ########## UPDATE ##############
            ego_vel = world.player.get_velocity()
            ego_loc = world.player.get_location()
            target_vehicle_loc = target_vehicle.get_location()
            target_vehicle_vel = target_vehicle.get_velocity()
            target_vehicle_vel = normalizer(target_vehicle_vel)

            # # Calculate velocity from the simulation
            ego_vel_transform = math.sqrt(
                ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
            distance_between_vehicles = compute_distance(
                target_vehicle_loc, ego_loc)

            i = i + 1
            if i == time_steps-1:
                print('Exceed itertation')
                break

    finally:
        if world is not None:
            world.destroy()
        recorder.end_recording()
        pygame.quit()
        # os.system(
        #    "avconv -r 8 -f image2 -i Snaps/%04d.png -y -qscale 0 -s 1280x720 -aspect 4:3 result.avi")
        T = int(hud.simulation_time)
        dt = T/size
        state_control(ut, dt, T)
        state_velocity(xt, dt, T)
        state_velocity_leading(xt, dt, T)
        state_relative_distance(xt, dt, T, kf_x, true_x)


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='192.168.10.243',  # '127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)


if __name__ == '__main__':

    main()
