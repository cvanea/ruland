import logging
from random import uniform, randint
from typing import List, Tuple
import os
import math

import numpy as np
import pygame

MUTATION_RATE = 0.05
MAP_SIZE = (1000, 1000)

logger = logging.getLogger(__name__)


class Value:
    def __init__(self, value: float | int):
        self.value = value

    def __repr__(self):
        return f"Value({self.value})"

    def __add__(self, other: "Value"):
        return Value(self.value + other.value)

    def __mul__(self, other: "Value"):
        return Value(self.value * other.value)

    def relu(self) -> "Value":
        return Value(max(0, self.value))

    def mutate(self):
        self.value *= uniform(-MUTATION_RATE, MUTATION_RATE)


class Neuron:
    def __init__(self, nin: int, is_output: bool = False):
        self.is_output = is_output
        self.weights = [Value(uniform(-1, 1)) for _ in range(nin)]

    def __repr__(self):
        return f"Neuron({self.weights})"

    def __call__(self, xs: List[Value]) -> Value:
        activation = Value(0.0)
        for w, x in zip(self.weights, xs):
            activation += w * x

        if self.is_output:
            return activation

        return activation.relu()

    def mutate(self):
        for w in self.weights:
            w.mutate()


class Layer:
    def __init__(self, nin: int, nout: int, is_output: bool = False):
        self.neurons = [Neuron(nin, is_output=is_output) for _ in range(nout)]

    def __repr__(self):
        return f"Layer({self.neurons})"

    def __call__(self, x: List[Value]) -> List[Value]:
        return [n(x) for n in self.neurons]

    def mutate(self):
        for n in self.neurons:
            n.mutate()


class Gene:
    def __init__(self):
        self.layers = [
            Layer(1, 4),
            Layer(4, 4),
            Layer(4, 4, is_output=True),
        ]

    def __call__(self, x: List[Value]) -> List[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def clone(self) -> "Gene":
        g = Gene()
        for layer in g.layers:
            layer.mutate()
        return g


class Critter:
    def __init__(self, gene: Gene, x: int = 0, y: int = 0):
        self.id = randint(0, 1000000000)
        self.gene = gene
        self.position = pygame.Vector2(x, y)

    def reproduce(self, env: "Environment"):
        # Create a new critter with a mutated gene and put it in the
        # environment.
        child = Critter(self.gene.clone(), self.position.x, self.position.y)
        env.critters[child.id] = child

    def move(self, env: "Environment"):
        activations = self.gene([Value(light(self.position.x, self.position.y))])
        [up, down, left, right] = [a.value for a in activations]

        max_value = max(up, down, left, right)
        speed = 10
        if max_value == up:
            self.position.y += speed
        elif max_value == down:
            self.position.y -= speed
        elif max_value == left:
            self.position.x -= speed
        elif max_value == right:
            self.position.x += speed

        if self.position.y < 0:
            env.remove(self)
        if self.position.y > MAP_SIZE[1]:
            env.remove(self)

    def act(self, env: "Environment"):
        # LEARNING: this is evaluated every tick, which is every ~16ms.
        # This means that even a pretty low death probability happens quite fast.
        df = death_factor(self.position.x, self.position.y)
        dies = uniform(0, 1) <= df
        if dies:
            logger.debug(f"Critter {self.id} died. RIP.")
            env.remove(self)
            return

        reproduces = uniform(0, 1) < 0.15
        if reproduces:
            logger.debug(f"Critter {self.id} reproduced. Congrats.")
            return self.reproduce(env)

        self.move(env)


class Environment:
    def __init__(self):
        self.critters = {}
        for _ in range(100):
            critter = Critter(Gene(), randint(0, MAP_SIZE[0]), randint(0, MAP_SIZE[1]))
            self.critters[critter.id] = critter

    def remove(self, critter: Critter):
        del self.critters[critter.id]

    def __repr__(self):
        return f"Environment({self.critters})"

def death_factor(x, y) -> float:
    return 1 - light(x, y)

def light(x, y) -> float:
    return 1 / (1.001 ** (_distance_to_centre(x, y)))

def _distance_to_centre(x, y) -> float:
    return math.sqrt(
        (x - MAP_SIZE[0] / 2) ** 2 + (y - MAP_SIZE[1] / 2) ** 2
    )

def gray(im):
    im = 255 * (im / im.max())
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

if __name__ == "__main__":
    env = Environment()

    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode(MAP_SIZE)
    clock = pygame.time.Clock()
    running = True
    dt = 0

    x = np.arange(0, 1000)
    y = np.arange(0, 1000)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(light)(X, Y)
    Z = gray(Z)

    surf = pygame.surfarray.make_surface(Z)
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    iteration = 0
    while running and iteration <= 10000:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                # fill the screen with a color to wipe away anything from last frame
                screen.blit(surf, (0, 0))

                critters = list(env.critters.values())
                if len(critters) == 0:
                    break
                for critter in critters:
                    pygame.draw.circle(screen, "red", critter.position, 5)
                    critter.act(env)

                # flip() the display to put your work on screen
                pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000
        iteration += 1

    pygame.quit()
