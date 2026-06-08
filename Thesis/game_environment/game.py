import pygame          # Library for game development (graphics, input handling)
import random          # Used for generating random pipe positions
import csv             # Used for storing gameplay data
import time            # Used for time measurement (duration, click intervals)
import os              # Used for file and directory management


# Game configuration constants
WIDTH, HEIGHT = 400, 600          # Window dimensions in pixels
GRAVITY = 0.25                    # Downward acceleration applied to the bird
FLAP_STRENGTH = -5                # Upward impulse when the player presses the key
PIPE_SPEED = 4                    # Speed at which obstacles move horizontally


class FlappyBirdClone:

    def __init__(self):
        pygame.init()  # Initialize all pygame modules
        
        # Create game window
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Bachelor's thesis: Behaviour Detection")
        
        self.clock = pygame.time.Clock()  # Controls frame rate
        self.font = pygame.font.SysFont("Arial", 22)  # Font for UI
        
        # Player-related variables
        self.player_name = ""
        self.round_counter = 0
        self.last_death_time = time.time()  # Used to calculate ITI
        
        # Initialize game
        self.get_name_input()
        self.reset_game()


    def get_name_input(self):
        """
        Displays an input screen where the player enters their name.
        This is required to associate collected data with a specific user.
        """
        
        input_active = True
        
        while input_active:
            self.screen.fill((50, 50, 50))  # Background color
            
            # Render text elements
            prompt_surface = self.font.render("Enter your name:", True, (255, 255, 255))
            name_surface = self.font.render(self.player_name + "_", True, (255, 255, 0))
            hint_surface = self.font.render("Press ENTER to start", True, (200, 200, 200))
            
            # Draw text on screen
            self.screen.blit(prompt_surface, (WIDTH//2 - 100, HEIGHT//2 - 60))
            self.screen.blit(name_surface, (WIDTH//2 - 100, HEIGHT//2 - 20))
            self.screen.blit(hint_surface, (WIDTH//2 - 120, HEIGHT//2 + 40))
            
            for event in pygame.event.get():
                
                # Exit application
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                
                # Handle keyboard input
                if event.type == pygame.KEYDOWN:
                    
                    if event.key == pygame.K_RETURN:
                        # Start game only if name is not empty
                        if len(self.player_name) > 0:
                            input_active = False
                    
                    elif event.key == pygame.K_BACKSPACE:
                        # Remove last character
                        self.player_name = self.player_name[:-1]
                    
                    else:
                        # Add character if length limit not exceeded
                        if len(self.player_name) < 15:
                            self.player_name += event.unicode
            
            pygame.display.update()
            self.clock.tick(30)


    def reset_game(self):
        """
        Resets all game variables for a new round.
        Also calculates the inter-trial interval (ITI).
        """
        
        current_time = time.time()
        self.iti = current_time - self.last_death_time  # Time between rounds
        
        # Initialize player (bird)
        self.bird = pygame.Rect(50, HEIGHT//2, 30, 30)
        self.bird_movement = 0
        
        # Reset obstacles and score
        self.pipes = []
        self.score = 0
        
        # Update round counter
        self.round_counter += 1
        
        # Initialize behavioral tracking
        self.start_time = time.time()
        self.clicks = 0
        self.click_times = []
        
        # Spawn first obstacle
        self.spawn_pipe()


    def spawn_pipe(self):
        """
        Generates a new pair of pipes with a random vertical gap.
        """
        
        gap_y = random.randint(150, 450)
        
        self.pipes.append(pygame.Rect(WIDTH, 0, 50, gap_y - 50))
        self.pipes.append(pygame.Rect(WIDTH, gap_y + 50, 50, HEIGHT))


    def calculate_features(self, death_y):
        """
        Calculates behavioral features for one game round.
        These features are later used as input for the ML model.
        """
        
        duration = time.time() - self.start_time  # Total round duration
        
        # Calculate time intervals between clicks
        intervals = [t2 - t1 for t1, t2 in zip(self.click_times, self.click_times[1:])]
        
        avg_int = sum(intervals) / len(intervals) if intervals else 0
        
        # Variance of intervals (measures irregularity)
        var_int = sum((x - avg_int) ** 2 for x in intervals) / len(intervals) if len(intervals) > 1 else 0
        
        # Count clicks in the last 2 seconds (panic behavior)
        recent_clicks = [t for t in self.click_times if t > time.time() - 2]
        panic_rate = len(recent_clicks)
        
        # Relative height at death
        death_altitude = round(death_y / HEIGHT, 2)

        return [
            round(duration, 2),
            self.score,
            self.clicks,
            round(avg_int, 3),
            round(var_int, 4),
            round(self.iti, 2),
            panic_rate,
            death_altitude
        ]


    def log_to_csv(self, features, label):
        """
        Saves the collected features and the player label
        (frustrated or engaged) into a CSV file.
        """
        
        file_path = 'data/player_data.csv'
        
        headers = [
            "player", "duration", "score", "clicks",
            "avg_interval", "var_interval", "iti",
            "panic_rate", "death_altitude", "label"
        ]
        
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header only once
            if not file_exists:
                writer.writerow(headers)
            
            # Write new data row
            writer.writerow([self.player_name] + features + [label])


    def show_feedback_screen(self):
        """
        Displays a feedback screen where the player manually
        labels their emotional state.
        """
        
        self.screen.fill((30, 30, 30))
        
        txt = self.font.render(
            f"Player: {self.player_name} - [F]rustrated or [E]ngaged?",
            True, (255, 255, 255)
        )
        
        self.screen.blit(txt, (15, HEIGHT//2))
        pygame.display.update()
        
        while True:
            for event in pygame.event.get():
                
                if event.type == pygame.KEYDOWN:
                    
                    if event.key == pygame.K_f:
                        return "frustrated"
                    
                    if event.key == pygame.K_e:
                        return "engaged"
                
                if event.type == pygame.QUIT:
                    return None


    def run(self):
        """
        Main game loop handling input, physics, rendering,
        and data collection.
        """
        
        running = True
        
        while running:
            
            self.screen.fill((135, 206, 235))  # Background color
            
            # --- Input handling ---
            for event in pygame.event.get():
                
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    
                    # Apply upward movement
                    self.bird_movement = FLAP_STRENGTH
                    
                    # Record click timestamp
                    self.click_times.append(time.time())
                    self.clicks += 1
            
            # --- Game physics ---
            self.bird_movement += GRAVITY
            self.bird.y += self.bird_movement
            
            # Move pipes
            for p in self.pipes:
                p.x -= PIPE_SPEED
            
            # Remove old pipes and increase score
            if self.pipes and self.pipes[0].x < -50:
                self.pipes.pop(0)
                self.pipes.pop(0)
                self.spawn_pipe()
                self.score += 1
            
            # --- Collision detection ---
            if (
                self.bird.top <= 0 or
                self.bird.bottom >= HEIGHT or
                any(self.bird.colliderect(p) for p in self.pipes)
            ):
                
                self.last_death_time = time.time()
                
                # Calculate features for ML model
                feat = self.calculate_features(self.bird.y)
                
                # Get user feedback (label)
                lbl = self.show_feedback_screen()
                
                if lbl:
                    self.log_to_csv(feat, lbl)
                
                # Start new round
                self.reset_game()
            
            # --- Rendering ---
            pygame.draw.rect(self.screen, (255, 255, 0), self.bird)
            
            for p in self.pipes:
                pygame.draw.rect(self.screen, (0, 255, 0), p)
            
            score_surface = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_surface, (10, 10))
            
            round_surface = self.font.render(f"Round: {self.round_counter}", True, (255, 255, 255))
            self.screen.blit(round_surface, (WIDTH - 120, 10))
            
            pygame.display.update()
            self.clock.tick(60)


# Entry point of the program
if __name__ == "__main__":
    
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Start the game
    FlappyBirdClone().run()