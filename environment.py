import pygame
import random
import numpy as np


pygame.init()
screenWidth, screenHeight = 300, 600 
screen = pygame.display.set_mode((screenWidth, screenHeight))
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

carWidth, carHeight = 50, 50
laneWidth = screenWidth // 3
lanes = [laneWidth // 2, laneWidth // 2 + laneWidth, laneWidth // 2 + 2 * laneWidth]

class AgentCar:
    def __init__(self):
        self.lane = 1
        self.y = screenHeight - 100
        self.rect = pygame.Rect(lanes[self.lane] - carWidth // 2, self.y, carWidth, carHeight)
        self.cooldown = 0  # Frames since last lane switch
        self.cooldownPeriod = 10  # Frames to wait before next lane switch (0.167s at 60 FPS)
    
    def move(self, action):
         # Decrease cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return  # No lane change during cooldown
        
        if action == 1 and self.lane > 0:
            self.lane -= 1
            self.cooldown = self.cooldownPeriod  # Reset cooldown
        elif action == 2 and self.lane < 2:
            self.lane += 1
            self.cooldown = self.cooldownPeriod  # Reset cooldown
        self.rect.x = lanes[self.lane] - carWidth // 2

class EnemyCar:
    def __init__(self, lane, speed):
        self.lane = lane
        self.y = 0
        self.speed = speed
        self.rect = pygame.Rect(lanes[self.lane] - carWidth // 2, self.y, carWidth, carHeight)

    def update(self):
        self.y += self.speed
        self.rect.y = self.y


class Environment:
    def __init__(self):
        self.agent = AgentCar()
        self.enemyCars = []
        self.spawnProbability = 0.02
        self.minSpeed = 2
        self.maxSpeed = 5
        self.maxEnemyCars = 5 
        self.font = pygame.font.SysFont('arial', 24)

    def reset(self):
        self.agent = AgentCar()
        self.enemyCars = []
        return self.getState()

    def step(self, action):
        # update agents position
        self.agent.move(action)
        #update enemy cars
        if action in [1, 2]:
            targetLane = self.agent.lane + (-1 if action == 1 else 1)
            sideCollision = any(car.lane == targetLane and abs(car.y - self.agent.y) <= 50 
                                for car in self.enemyCars)
            if sideCollision:
                reward = -100
                done = True

        for car in self.enemyCars:
            car.update()
        self.enemyCars = [car for car in self.enemyCars if car.y < screenHeight]

        if len(self.enemyCars) < self.maxEnemyCars and random.random() < self.spawnProbability:
            lane = random.randint(0, 2)
            speed = random.uniform(self.minSpeed, self.maxSpeed)
            self.enemyCars.append(EnemyCar(lane, speed))

        # check collision
        collision = any(car.lane == self.agent.lane and abs(car.y - self.agent.y) < carHeight for car in self.enemyCars)
        reward = -100 if collision else 1
        done = collision 
        nextState = self.getState()
        return nextState, reward, done
    
    def getState(self):
        myLane = self.agent.lane 
        agentY = self.agent.y
        #distsAbove = []
        #speedsAbove = []
        #distsSide = [1000, 1000, 1000]  # Default for left, middle, right lanes
        #speedsSide = [0, 0, 0]  # Default for left, middle, right lanes
        distsAhead = [200] * 3
        speedsAhead = [0] * 3
        """for lane in range(3):
            carsAbove = [car for car in self.enemyCars if car.lane == lane and car.y < agentY]
            if carsAbove:
                nearest = max(carsAbove, key=lambda c: c.y)
                distsAbove.append(agentY - nearest.y)
                speedsAbove.append(nearest.speed)
            else:
                distsAbove.append(1000)
                speedsAbove.append(0)

        # cars to the side
        for lane in range(3):
            carsSide = [car for car in self.enemyCars if car.lane == lane and abs(car.y - agentY) <= 50]

            if carsSide:
                nearest = min(carsSide, key=lambda c: abs(c.y - agentY))
                distsSide[lane] = abs(agentY - nearest.y)
                speedsSide[lane] = nearest.speed"""

        # Detect closest car in each lane ahead (y <= agentY, within 300 pixels)
        for lane in range(3):
            carsAhead = [car for car in self.enemyCars if car.lane == lane and 
                          car.y <= agentY and (agentY - car.y) <= 300]
            if carsAhead:
                nearest = max(carsAhead, key=lambda c: c.y)  # Closest car (highest y)
                distsAhead[lane] = agentY - nearest.y
                speedsAhead[lane] = nearest.speed
        
        # Normalise distances
        distsAhead = [min(d / 300, 1) for d in distsAhead]

        # State: [lane_onehot (3), distsAhead (3), speedsAhead (3)]
        state = [1 if i == myLane else 0 for i in range(3)] + distsAhead + speedsAhead
        return np.array(state)

        #state = [1 if i == myLane else 0 for i in range(3)] + distsAbove + speedsAbove + distsSide + speedsSide
        #return np.array(state)
    
    def render(self, episode=0, totalReward=0):
        screen.fill(WHITE)
        # Draw Lanes
        for i in range(1, 3):
            pygame.draw.line(screen, BLACK, (i * laneWidth, 0), (i * laneWidth, screenHeight))
        
        # Draw Cars
        pygame.draw.rect(screen, RED, self.agent.rect)
        for car in self.enemyCars:
            pygame.draw.rect(screen, BLACK, car.rect)

        # Draw lines to detected cars (same logic as get_state)
        agentY = self.agent.y
        lineColours = ['BLUE', 'GREEN', 'CYAN']  # Left, middle, right lanes
        for lane in range(3):
            carsAhead = [car for car in self.enemyCars if car.lane == lane and 
                          car.y <= agentY and (agentY - car.y) <= 300]
            if carsAhead:
                nearest = max(carsAhead, key=lambda c: c.y)  # Closest car
                # Draw line from agent's center to car's center
                startPos = (lanes[self.agent.lane], agentY)
                endPos = (lanes[nearest.lane], nearest.y)
                pygame.draw.line(screen, lineColours[lane], startPos, endPos, 2)
        
        episodeText = self.font.render(f'Episode: {episode}', True, BLACK)
        rewardText = self.font.render(f'Reward: {totalReward}', True, BLACK)
        screen.blit(episodeText, (10, 10))
        screen.blit(rewardText, (10, 40))
        pygame.display.flip()
        pygame.display.flip()

    def close():
        pygame.quit()