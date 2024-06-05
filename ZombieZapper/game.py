import pygame
import sys
import random
import settings
import config


class Game:
    def __init__(self) -> None:
        '''Initialize the game'''
        pygame.init()
        self.time = pygame.time.Clock()
        self.frames_per_second = 60 
        self.game_over = False
        
        config.DisplayManager(settings.WIDTH, settings.HEIGHT)
        config.EnvironmentManager(settings.WIDTH, settings.HEIGHT)
        self.sound = config.SoundManager()
        self.sound.sound_setting()
        
        self.display = config.DisplayManager(settings.WIDTH, settings.HEIGHT)
        self.enviourment = config.EnvironmentManager(settings.WIDTH, settings.HEIGHT)
        self.bullet_setting = config.BulletAnimation()
        self.player = config.Player()
        self.player_group = pygame.sprite.Group()
        self.player_group.add(self.player)
        
        self.spawn_animations = pygame.sprite.Group()
        self.all_enemies_group = pygame.sprite.Group()
        
        self.score = 0
        self.in_menu = True
        
        self.enemy_speed = settings.STARTING_ENEMY_SPEED  
        self.elapsed_time = 0  
        self.speed_up_interval = 5000 
        
    def show_menu(self):
        self.menu = pygame.transform.scale(pygame.image.load(r'D:\Python Mini Projects\ZombieZapper\Game_Assets\4. Background Art\full_menu.jpg'), (settings.WIDTH, settings.HEIGHT)).convert_alpha()
        self.display.screen.blit(self.menu, (0,0))
        pygame.display.flip()
        
    def transition_to_game(self):
        self.in_menu = False
        
    def restart_game(self):
            # Reset variables or game state as needed
            self.score = 0
            self.game_over = False
            self.all_enemies_group.empty()
            self.spawn_animations.empty()
            self.player.rect.midbottom = (400, 640)
            self.enemy_speed = settings.STARTING_ENEMY_SPEED
            self.start_main_sound()
    
    def stop_main_sound(self):
        self.sound.stop_main_sound()
    
    def start_main_sound(self):
        self.sound.sound_setting()
        
    def run(self):
        running = True
        start_time = pygame.time.get_ticks()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.in_menu and event.key == pygame.K_SPACE:
                        self.transition_to_game()
                    elif not self.in_menu and event.key == pygame.K_p:
                        if self.game_over:
                            self.restart_game()
                            self.score = 0
                    elif not self.in_menu:
                        self.bullet_setting.fire(event, self.player)

            if self.in_menu:
                self.show_menu()
            
            elif not self.game_over:
                # Player Movement
                self.keys = pygame.key.get_pressed()
                self.player.update(self.keys)

                # Check for collisions with enemies
                for enemy in self.all_enemies_group:
                    if self.player.rect.colliderect(enemy.rect):
                        self.game_over = True
            
                self.bullet_setting.update(self.all_enemies_group, self.spawn_animations)
            
                if random.randint(0, 30) == 0:
                    self.new_enemy_type = random.choice([config.Enemy, config.AnotherEnemy])
                    self.new_enemy = self.new_enemy_type(settings.WIDTH, 645)
                    self.all_enemies_group.add(self.new_enemy)
                
                
                # Draw Background
                self.display.screen.blit(self.enviourment.background_layer_1, (0, 0))
                self.display.screen.blit(self.enviourment.background_layer_2, (0, 0))
                self.display.screen.blit(self.enviourment.background_layer_3, (0, 0))

                # Draw Foreground
                self.display.screen.blit(self.enviourment.scaled_foreground_layer_1, (1100, 220))
                self.display.screen.blit(self.enviourment.scaled_foreground_layer_2, (-65, 335))
                self.display.screen.blit(self.enviourment.scaled_foreground_layer_3, (0, 595))
                self.display.screen.blit(self.enviourment.scaled_foreground_layer_3, (371, 595))
                self.display.screen.blit(self.enviourment.scaled_foreground_layer_3, (742, 595))
                self.display.screen.blit(self.enviourment.scaled_foreground_layer_3, (1113, 595))
                self.display.screen.blit(self.enviourment.scaled_foreground_layer_3, (1484, 595))
                self.display.screen.blit(self.enviourment.scaled_foreground_layer_4, (850, 260))

                
                # Draw Bullets
                for bullet in self.bullet_setting.bullets:
                    self.display.screen.blit(self.bullet_setting.bullet_image, bullet)

                # Update and Draw Spawn Animations
                self.spawn_animations.update()
                self.spawn_animations.draw(self.display.screen)
                
                # Update and draw enemies
                self.elapsed_time = pygame.time.get_ticks() - start_time
                
                if self.elapsed_time > self.speed_up_interval:
                    self.enemy_speed += settings.ENEMY_SPEED_INCREASE
                    start_time = pygame.time.get_ticks()
                
                for enemy in self.all_enemies_group:
                    enemy.update(self.enemy_speed)
                    self.display.screen.blit(enemy.image, enemy.rect)
                
                # Increment the score when an enemy is killed
                new_bullets = []

                for bullet in self.bullet_setting.bullets:
                    bullet_hit_enemy = False

                    for enemy in self.all_enemies_group:
                        if bullet.colliderect(enemy.rect):
                            self.score += 1 
                            bullet_hit_enemy = True

                    if not bullet_hit_enemy:
                        new_bullets.append(bullet)

                # Update the bullets list with the new one
                self.bullet_setting.bullets = new_bullets

                self.font = pygame.font.Font(r'D:\Python Mini Projects\ZombieZapper\Game_Assets\font\BD_Cartoon_Shout.ttf', 25)
                self.score_text = self.font.render(f"Score: {self.score}", True, (0, 2, 10))
                self.score_rect = self.score_text.get_rect(center=(settings.WIDTH // 5, 100))
                self.display.screen.blit(self.score_text, self.score_rect)
    
                # Display Current Crypto Image
                self.display.screen.blit(self.enviourment.scaled_cryopod_images[self.enviourment.current_frame_crypto], dest=(910, 140))
                self.enviourment.current_frame = (self.enviourment.current_frame + 1) % 14
                self.enviourment.animation_count_crypto += self.enviourment.animation_speed_crypto
                self.enviourment.current_frame_crypto = (self.enviourment.animation_count_crypto // 60) % len(self.enviourment.crypto_images)
                
                # Draw Character and Gun
                self.display.screen.blit(self.player.character_image, self.player.rect)
                self.display.screen.blit(self.player.gun_image, self.player.gun_rect)
                
            else:  
                # Game over screen
                self.game_over = pygame.transform.scale(pygame.image.load(r'D:\Python Mini Projects\ZombieZapper\Game_Assets\4. Background Art\restart.jpg'), (settings.WIDTH, settings.HEIGHT)).convert_alpha()
                self.display.screen.blit(self.game_over, (0,0))
                
                self.font = pygame.font.Font(r'D:\Python Mini Projects\ZombieZapper\Game_Assets\font\BD_Cartoon_Shout.ttf', 45)
                self.score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
                self.score_rect = self.score_text.get_rect(center=(settings.WIDTH // 2, 193))
                self.display.screen.blit(self.score_text, self.score_rect)
                self.stop_main_sound()
                
            pygame.display.flip()
            self.time.tick(self.frames_per_second)

        pygame.quit()
        sys.exit()
        
if __name__ == '__main__':
    game = Game()
    game.run()
