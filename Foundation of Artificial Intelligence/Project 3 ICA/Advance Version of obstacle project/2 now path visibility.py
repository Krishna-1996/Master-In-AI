# Main game loop
def game_loop():
    global running
    while running:
        handle_input()
        draw_grid()  # This will keep the grid and obstacles updated

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Press 'r' to reset the maze
                    reset_maze()
                elif event.key == pygame.K_SPACE:  # Press 'space' to run all algorithms
                    run_all_algorithms()

        clock.tick(30)  # Control the frame rate

    pygame.quit()
    root.quit()
