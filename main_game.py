import pygame
import torch
from pong_game_objects import *



if __name__ == '__main__':

    input("""
            Pong main game script:

            Use:    - Run script and follow input prompts to setup windowsize and players.
                      Current Options: Human, Perfect, NN

                    - Paddle 2 also has the option 'None' which allows single player
                      play with the left side paddle against a rebounding wall.

                    - Option for logging date for all active paddles.

                    - During main game loop, the game will reset if the ball falls below a 
                      defined speed and/or angle.  (For training purposes, can be removed)
                    
                    - The <Return> key will reset the ball position of the ball at anytime.

                    - Closing the screen will exit the game and close the files in use.
          
          

          Press <Return> to begin.

          """)


    # Set display width and height
    Width  = int(input('\nInput screen Width:    '))
    Height = int(input('\nInput screen Height:   '))


    # Initialize Player 1
    while True:
        player_1 = input('\n\nPlayer 1: Human, Perfect or NN?     ')
        if player_1 in ('Human', 'Perfect', 'NN'):
            break
        else:
            print("\n INVALID INPUT : Must be in ('Human', 'Perfect', 'NN') \n")

    if player_1 == 'Human':
        paddle_1 = Player_Paddle(Width, Height, 1)
    elif player_1 == 'Perfect':
        paddle_1 = Perfect_Paddle(Width, Height, 1)
    elif player_1 == 'NN':
        Agent_1 = torch.load('NN Weights/Pytorch_1')
        Agent_1.eval()
        paddle_1 = NeuralNetwork_Paddle(Width, Height, Agent_1, 1)


    # Initialize Player 2
    while True:
        player_2 = input('\n\nPlayer 2: Human, Perfect, NN, or None?     ')
        if player_2 in ('Human', 'Perfect', 'NN', 'None'):
            break
        else:
            print("\n INVALID INPUT : Must be in ('Human', 'Perfect', 'NN') \n")

    if player_2 == 'Human':
        paddle_2 = Player_Paddle(Width, Height, 2)
    elif player_2 == 'Perfect':
        paddle_2 = Perfect_Paddle(Width, Height, 2)
    elif player_2 == 'NN':
        Agent_2 = torch.load('NN Weights/Pytorch_1')
        Agent_2.eval()
        paddle_2 = NeuralNetwork_Paddle(Width, Height, Agent_2, 2)
    
    
    # Total players for setting ball properties
    if player_2 == 'None':
        total_players = 1
    else:
        total_players = 2


    # Setup files for data logging
    while True:
        log_data = input('\n\nLog Data?     ')
        if log_data in ('Y', 'N'):
            if log_data == 'Y':
                file_1 = open('Training Data/paddle_data_left.txt', 'a')
                if total_players == 2:
                    file_2 = open('Training Data/paddle_data_right.txt', 'a')
            break
        else:
            print("\n INVALID INPUT : Must be in ('Y', 'N') \n")


    # Initialize Ball
    ball = Ball(Width, Height, total_players)



    # Setup game screen and define colors
    pygame.init()

    screen = pygame.display.set_mode((Width, Height))

    bgColor = pygame.Color("black")
    fgColor = pygame.Color("green")


    # Initialize clock and set framerate
    clock = pygame.time.Clock()
    framerate = 60

    """
    Main Game Loop
    """
    while True:
        e = pygame.event.poll()
        if e.type == pygame.QUIT:
            break
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_RETURN:
                ball.reset_ball()


        pygame.display.flip()


        # Reset  ball if moving too slow
        if abs(ball.vy) < ball.V_min or abs(ball.vx) < ball.V_min:
            ball.reset_ball()


        Vx_ball = ball.vx/ball.V_max
        Vy_ball = ball.vy/ball.V_max


        screen.fill(bgColor)


        # Update left side paddle
        if player_1 == 'Human':
            paddle_1.update(screen, bgColor, fgColor)
        elif player_1 == 'Perfect':
            paddle_1.update(screen, bgColor, fgColor, ball, paddle_1.y)
        elif player_1 == 'NN':
            X = torch.tensor([ball.x/Width, ball.y/Height, Vx_ball, Vy_ball], dtype=torch.float)
            paddle_1.update(screen, bgColor, fgColor, X, paddle_1.y)

        
        # Update right side paddle
        if player_2 == 'Human':
            paddle_2.update(screen, bgColor, fgColor)
        elif player_2 == 'Perfect':
            paddle_2.update(screen, bgColor, fgColor, ball, paddle_2.y)
        elif player_2 == 'NN':
            X = torch.tensor([ball.x/Width, ball.y/Height, Vx_ball, Vy_ball], dtype=torch.float)
            paddle_2.update(screen, bgColor, fgColor, X, paddle_2.y)


        if total_players == 1:
            ball.update(paddle_1, screen, bgColor, fgColor)
        else:
            # If there are 2 players, switch between paddles as ball moves to each side
            if ball.x < Width//2:
                ball.update(paddle_1, screen, bgColor, fgColor)
            else:
                ball.update(paddle_2, screen, bgColor, fgColor)


        # Log data if setup
        if log_data == 'Y':
            file_1.write('%f %f %f %f %f\n' % (ball.x/Width, ball.y/Height, Vx_ball, Vy_ball, paddle_1.y/Height))
            if total_players == 2:
                file_2.write('%f %f %f %f %f\n' % (ball.x/Width, ball.y/Height, Vx_ball, Vy_ball, paddle_2.y/Height))


        clock.tick(framerate)

    # Preform cleanup after quitting
    if log_data == 'Y':
        file_1.close()
        if total_players == 2:
            file_2.close()

    pygame.quit()