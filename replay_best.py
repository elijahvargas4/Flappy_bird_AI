import pygame
import neat
import pickle
import os
from main import Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, draw_window
import main
main.DRAW_LINES = False

pygame.font.init()

gen = 0
# Load genome and config
def load_genome(config_path):
    with open("best_genome.pkl", "rb") as f:
        genome = pickle.load(f)
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    return genome, config

def replay_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    bird = Bird(230, 350)
    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    score = 0
    run = True
    while run:
        clock.tick(500)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        # Move bird
        bird.move()

        # Use neural network to decide jump
        pipe_ind = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
            pipe_ind = 1

        output = net.activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

        if output[0] > 0.5:
            bird.jump()

        # Move environment
        add_pipe = False
        rem = []
        for pipe in pipes:
            pipe.move()
            if pipe.collide(bird, win):
                run = False
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
            run = False

        base.move()
        
        draw_window(win, [bird], pipes, base, score, gen, pipe_ind)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_feedforward.txt")
    genome, config = load_genome(config_path)
    replay_genome(genome, config)