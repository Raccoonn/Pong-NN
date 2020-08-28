
import pygame
import random
import pong_NN
import numpy as np

"""
Update pong_NN to use biases instead of a scaling factor.  I realize now that the scaling factor
was probaly the source of the height output issues.

Also set the network to save during each epoch of the training.

This program also loads the weights from a file even though they are initialized when the network
is created.  Basically the load is just overwriting the random initialization values.


Running a network with bias instead of scaling factor has yeilded good results for the code abbey
problem.  However with the same amount of trainin with the proper scaling factor does yield better
results.  Possibly need more time to now also train biases.
NOTE: That is also a very small and arbitrary data set.
"""


class Ball:
    """
    Object for the moving ball.
        - Better understand the calls to paddle.y in the update method.
    """
    Radius = 10
    # V_max = 20
    # V_min = 8

    V_max = 20

    def __init__(self, x, y):
        self.x = x
        self.y = y

        # self.Speeds = [n for n in range(-self.V_max, self.V_max+1) if n not in list(range(-self.V_min, self.V_min+1))]
        
        self.Speeds = [self.V_max, -self.V_max]

        self.vx = random.choice(self.Speeds)
        self.vy = random.choice(self.Speeds)



    def reset_ball(self, Height, Width):
        self.x = np.random.randint(1, Width//3)
        self.y = np.random.randint(1, Height-1)

        self.vx = random.choice(self.Speeds)
        self.vy = random.choice(self.Speeds)



    def show(self, color):
        """ Draw the ball on screen """
        global screen
        pygame.draw.circle(screen, color, (self.x, self.y), self.Radius)



    def update(self, y_paddle, vy_paddle):
        global bgColor, fgColor

        new_x = self.x + self.vx
        new_y = self.y + self.vy

        # Left Side
        if new_x < self.Radius:
            # self.vx = -round(self.vx * 0.9)
            self.vx *= -1

        # Top and Bottom
        elif new_y < self.Radius or new_y > Height - self.Radius:
            # self.vy = -round(self.vy * 0.9)
            self.vy *= -1

        # Contact with paddle
        elif new_x + self.Radius >  Width - Paddle.Width and abs(new_y - y_paddle) < Paddle.Height//2:
            # self.vx = -1 * ( self.vx + abs(vy_paddle//3) )
            # self.vy = self.vy + abs(vy_paddle//2)
            self.vx *= -1

        elif new_x > Width - self.Radius:
            self.reset_ball(Height, Width)
        else:
            self.show(bgColor)
            self.x += self.vx
            self.y += self.vy
            self.show(fgColor)



class Paddle:
    """
    Paddle object.
        - Currently only one paddle located on the right side of the screen.
    """
    Width = 20
    Height = 100

    def __init__(self, y, player):
        self.y = y
        self.vy = 0
        self.player = player

    
    def show(self, color):
        """
        Show object on screen (Drawing object rather than blitting an image).
        """
        global screen
        if self.player == 1:
            pygame.draw.rect(screen, color, pygame.Rect((0, self.y - self.Height//2, self.Width, self.Height)))
        elif self.player == 2:
            pygame.draw.rect(screen, color, pygame.Rect((Width - self.Width, self.y - self.Height//2, self.Width, self.Height)))


    def update(self):
        """
        Update positon but taking the Y coordinate of the mouse.
        Only using y so x is set to screen Width - Paddle Width
        """
        self.show(pygame.Color("black"))
        self.y = pygame.mouse.get_pos()[1]
        self.vy = pygame.mouse.get_rel(1200-self.Width, self.y)[1]
        self.show(pygame.Color("white"))



class Perfect_Paddle:
    """
    Paddle that hits the ball everytime.
    """
    Width = 20
    Height = 100

    def __init__(self, y):
        self.y = y
        self.vy = 0


    def show(self, color):
        """
        Show object on screen.
        """
        global screen
        pygame.draw.rect(screen, color, pygame.Rect((Width-self.Width, self.y-self.Height//2, self.Width, self.Height)))


    def update(self, ball):
        """
        Update paddle position from Neural Network by doing feedForward on X.
        """
        self.show(pygame.Color("black"))

        self.y = ball.y

        self.show(pygame.Color("white"))
   






class NN_Paddle:
    """
    Paddle to be conmtrolled by trained neural network.
    """
    Width = 20
    Height = 100

    def __init__(self, y, NN):
        self.y = y
        self.vy = 0
        self.agent = NN


    def show(self, color):
        """
        Show object on screen.
        """
        global screen
        pygame.draw.rect(screen, color, pygame.Rect((Width-self.Width, self.y-self.Height//2, self.Width, self.Height)))


    def update(self, X):
        """
        Update paddle position from Neural Network by doing feedForward on X.
        """
        self.show(pygame.Color("black"))
        
        # Get y value from Neural Network
        self.y = self.agent.feedForward(X) * Height

        if self.y < 0:
            self.y = 0
        elif self.y > Height:
            self.y = Height

        self.show(pygame.Color("white"))






def write_data(file, x, y, vx, vy, y_paddle_frac):
    """
    Function to store data for training later.
    y_paddle_frac is the paddles normalized height
    """
    file.write('%f %f %f %f %f\n' % (x, y, vx, vy, y_paddle_frac))









if __name__ == '__main__':

    log_data = input('\n\nLog Data?    ')
    if log_data == 'Y':
        file = open('pong_training.txt', 'w')


    load_network = input('\n\nLoad Neural Network?    ')
    if load_network == 'Y':
        # Load Neural Network
        K = 4
        NN = pong_NN.NeuralNetwork(K)
        NN.load_weights('2_weights_Best.txt')

    pygame.init()


    Width, Height = 1200, 600

    Velocity = 15

    screen = pygame.display.set_mode((Width, Height))

    fgColor = pygame.Color("white")
    bgColor = pygame.Color("black")

    screen.fill(bgColor)




    ball = Ball(np.random.randint(1, Width//3), np.random.randint(1, Height-1))
    ball.show(fgColor)



    # Currently swap out AI and player paddle

    # paddle = NN_Paddle(Height//2, NN)
    paddle = Perfect_Paddle(Height//2)
    # paddle = Paddle(Height//2, 2)

    paddle.show(fgColor)


    clock = pygame.time.Clock()


    while True:
        e = pygame.event.poll()
        if e.type == pygame.QUIT:
            break
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_RETURN:
                ball.reset_ball(Height, Width)


        pygame.display.flip()

        if log_data == 'Y':
            write_data(file, ball.x/Width, ball.y/Height, ball.vx/ball.V_max, ball.vy/ball.V_max, paddle.y/Height)

        screen.fill(bgColor)

        # Currently swap out AI and player paddle

        # paddle.update(np.array([ball.x/Width, ball.y/Height, ball.vx/ball.V_max, ball.vy/ball.V_max]))
        paddle.update(ball)
        # paddle.update()

        ball.update(paddle.y, paddle.vy)

        clock.tick(60)



    pygame.quit()




