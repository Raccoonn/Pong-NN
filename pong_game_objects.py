import pygame
import random



"""
Ball Object:
    - Contains methods for updating the ball each frame, handling collisions,
      displaying the ball on the screen, and to reset the ball.
"""

class Ball:
    Radius = 10

    V_max = 30
    V_min = 5

    def __init__(self, Width, Height, total_players):
        self.screen_Width = Width
        self.screen_Height = Height

        self.total_players = total_players

        self.x = self.screen_Width//2
        self.y = random.randint(1, self.screen_Height-1)

        self.Speeds = [n for n in range(-self.V_max, self.V_max+1) if n not in list(range(-self.V_max//2, self.V_max//2))]

        self.vx = random.choice(self.Speeds)
        self.vy = random.choice(self.Speeds)



    def reset_ball(self):
        self.x = self.screen_Width//2
        self.y = random.randint(1, self.screen_Height-1)

        self.vx = random.choice(self.Speeds)
        self.vy = random.choice(self.Speeds)



    def update(self, paddle, screen, bgColor, fgColor):
        new_x = self.x + self.vx
        new_y = self.y + self.vy

        # Left side
        if new_x < self.Radius:
            self.reset_ball()

        # Top and Bottom
        elif new_y < self.Radius or new_y > self.screen_Height-self.Radius:
            self.vy = -round(self.vy * 0.9)

        # Contact with paddle
        elif new_x-self.Radius <  Player_Paddle.Width and abs(new_y-paddle.y) < Player_Paddle.Height//2:
            self.vx = -( self.vx + abs(paddle.vy//3) )
            self.vy += abs(paddle.vy//4)

        elif new_x+self.Radius >  self.screen_Width-Player_Paddle.Width and abs(new_y-paddle.y) < Player_Paddle.Height//2:
            self.vx = -( self.vx + abs(paddle.vy//3) )
            self.vy += abs(paddle.vy//4)


        # Right side
        elif new_x > (self.screen_Width-self.Radius):
            if self.total_players == 1:
                self.vx = -round(self.vx * 0.9)
            elif self.total_players == 2:
                self.reset_ball()
        
        
        pygame.draw.circle(screen, bgColor, (self.x, self.y), self.Radius)
        self.x += self.vx
        self.y += self.vy
        pygame.draw.circle(screen, fgColor, (self.x, self.y), self.Radius)
        



"""
Paddle Objects:
    - All paddles have the same structure; initialization and method to update the 
      paddle on each frame.

    - They differ in that each paddle takes a different input to operate.
    
    - All paddle update methods also take the screen, background color and foreground 
      color as inputs.
"""


class Player_Paddle:
    """
    Player's Paddle Object:  Uses mouse input for control
    """
    Width = 20
    Height = 100

    def __init__(self, Width, Height, player):
        self.screen_Width = Width
        self.screen_Height = Height

        self.player = player

        self.y = self.screen_Height//2
        self.vy = 0


    def update(self, screen, bgColor, fgColor):
        if self.player == 1:
            pygame.draw.rect(screen, bgColor, pygame.Rect((0, self.y-self.Height//2, self.Width, self.Height)))
        elif self.player == 2:
            pygame.draw.rect(screen, bgColor, pygame.Rect((self.screen_Width-self.Width, self.y-self.Height//2, self.Width, self.Height)))

        self.y = pygame.mouse.get_pos()[1]
        self.vy = pygame.mouse.get_rel(self.screen_Width-self.Width, self.y)[1]

        if self.player == 1:
            pygame.draw.rect(screen, fgColor, pygame.Rect((0, self.y-self.Height//2, self.Width, self.Height)))
        elif self.player == 2:
            pygame.draw.rect(screen, fgColor, pygame.Rect((self.screen_Width-self.Width, self.y-self.Height//2, self.Width, self.Height)))




class Perfect_Paddle:
    """
    Takes ball object as input and sets paddle.y == ball.y
    """
    Width = 20
    Height = 100

    def __init__(self, Width, Height, player):
        self.screen_Width = Width
        self.screen_Height = Height

        self.player = player
        
        self.y = self.screen_Width//2
        self.vy = 0


    def update(self, screen, bgColor, fgColor, ball, y_last):
        if self.player == 1:
            pygame.draw.rect(screen, bgColor, pygame.Rect((0, self.y-self.Height//2, self.Width, self.Height)))
        elif self.player == 2:
            pygame.draw.rect(screen, bgColor, pygame.Rect((self.screen_Width-self.Width, self.y-self.Height//2, self.Width, self.Height)))

        self.y = ball.y
        self.vy = round(self.y - y_last)

        if self.player == 1:
            pygame.draw.rect(screen, fgColor, pygame.Rect((0, self.y-self.Height//2, self.Width, self.Height)))
        elif self.player == 2:
            pygame.draw.rect(screen, fgColor, pygame.Rect((self.screen_Width-self.Width, self.y-self.Height//2, self.Width, self.Height)))




class NeuralNetwork_Paddle:
    """
    Takes ball position and velocity as a 4 element tensor preforms a feedforward and 
    returns paddle.y/Height fraction.

    Multiply by screen height to get paddle pixel position.
    """
    Width = 20
    Height = 100

    def __init__(self, Width, Height, Agent, player):
        self.screen_Width = Width
        self.screen_Height = Height
        
        self.player = player
        
        self.y = self.screen_Height//2
        self.vy = 0
        self.agent = Agent


    def update(self, screen, bgColor, fgColor, X, y_last):
        if self.player == 1:
            pygame.draw.rect(screen, bgColor, pygame.Rect((0, self.y-self.Height//2, self.Width, self.Height)))
        elif self.player == 2:
            pygame.draw.rect(screen, bgColor, pygame.Rect((self.screen_Width-self.Width, self.y-self.Height//2, self.Width, self.Height)))

        
        # Get y value from Neural Network
        self.y = float(self.agent.forward(X)) * self.screen_Height
        self.vy = int(self.y - y_last)

        if self.y < 0:
            self.y = 0
        elif self.y > self.screen_Height:
            self.y = self.screen_Height

        if self.player == 1:
            pygame.draw.rect(screen, fgColor, pygame.Rect((0, self.y-self.Height//2, self.Width, self.Height)))
        elif self.player == 2:
            pygame.draw.rect(screen, fgColor, pygame.Rect((self.screen_Width-self.Width, self.y-self.Height//2, self.Width, self.Height)))