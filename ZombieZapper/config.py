import pygame
import random


class DisplayManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode(size=(width, height))
        pygame.display.set_caption('Zombie Zapper')


class EnvironmentManager:
    def __init__(self, width, height):
        
        # Environment Setting
        self.background_layer_1 = pygame.transform.scale(pygame.image.load(r'D:\Python Mini Projects\ZombieZapper\Game_Assets\4. Background Art\Layer 3.PNG'), (width, height)).convert_alpha()
        self.background_layer_2 = pygame.transform.scale(pygame.image.load(r'D:\Python Mini Projects\ZombieZapper\Game_Assets\4. Background Art\Layer 2.PNG'), (width, height)).convert_alpha()
        self.background_layer_3 = pygame.transform.scale(pygame.image.load(r'D:\Python Mini Projects\ZombieZapper\Game_Assets\4. Background Art\Layer 1.PNG'), (width, height)).convert_alpha()

        self.foreground_layer_1 = pygame.image.load(r'D:\Python Mini Projects\ZombieZapper\Game_Assets\3. Foreground Art\Bunker.PNG').convert_alpha()
        self.foreground_layer_1_width = self.foreground_layer_1.get_width() // 1.6
        self.foreground_layer_1_height = self.foreground_layer_1.get_height() // 1.6
        self.scaled_foreground_layer_1 = pygame.transform.scale(self.foreground_layer_1, (self.foreground_layer_1_width, self.foreground_layer_1_height))

        self.foreground_layer_2 = pygame.image.load(r'D:\Python Mini Projects\ZombieZapper\Game_Assets\3. Foreground Art\Bus Stop.PNG').convert_alpha()
        self.foreground_layer_2_width = self.foreground_layer_2.get_width() // 1.4
        self.foreground_layer_2_height = self.foreground_layer_2.get_height() // 1.4
        self.scaled_foreground_layer_2 = pygame.transform.scale(self.foreground_layer_2, (self.foreground_layer_2_width, self.foreground_layer_2_height))

        self.foreground_layer_3 = pygame.image.load(r'D:\Python Mini Projects\ZombieZapper\Game_Assets\3. Foreground Art\Ground.PNG').convert_alpha()
        self.foreground_layer_3_width = self.foreground_layer_3.get_width() // 2
        self.foreground_layer_3_height = self.foreground_layer_3.get_height() // 2
        self.scaled_foreground_layer_3 = pygame.transform.scale(self.foreground_layer_3, (self.foreground_layer_3_width, self.foreground_layer_3_height))

        self.foreground_layer_4 = pygame.image.load(r'D:\Python Mini Projects\ZombieZapper\Game_Assets\3. Foreground Art\Platform.PNG').convert_alpha()
        self.foreground_layer_4_width = self.foreground_layer_4.get_width() // 2
        self.foreground_layer_4_height = self.foreground_layer_4.get_height() // 2
        self.scaled_foreground_layer_4 = pygame.transform.scale(self.foreground_layer_4, (self.foreground_layer_4_width, self.foreground_layer_4_height))

        # Load crypto images
        self.crypto_images = [pygame.image.load(fr'D:\Python Mini Projects\ZombieZapper\Game_Assets\3. Foreground Art\Cryo Pod\Cryo{i}.PNG').convert_alpha() for i in range(1, 14)]
        self.current_frame = 0 
        self.current_frame_crypto = 0
        self.animation_count_crypto = 0
        self.animation_speed_crypto = 10
        
        self.original_width, self.original_height = self.crypto_images[0].get_width(), self.crypto_images[0].get_height()
        self.scaled_width, self.scaled_height = int(self.original_width // 1.3), int(self.original_height // 1.3)
        self.scaled_cryopod_images = [pygame.transform.scale(img, (self.scaled_width, self.scaled_height)) for img in self.crypto_images]


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.run_frames = [pygame.image.load(fr"D:\Python Mini Projects\ZombieZapper\Game_Assets\1.MainCharacterArt\Animations\Run\Run{i}.PNG").convert_alpha() for i in range(1, 7)]
        self.jump_frames = [pygame.image.load(fr"D:\Python Mini Projects\ZombieZapper\Game_Assets\1.MainCharacterArt\Animations\Jump\Jump{i}.PNG").convert_alpha() for i in range(1, 3)]
        self.character_image = self.run_frames[0]
        self.image = self.character_image
        self.rect = self.image.get_rect(midbottom=(400, 640))
        self.current_frame = 0
        self.gun_image = pygame.image.load(r"D:\Python Mini Projects\ZombieZapper\Game_Assets\1.MainCharacterArt\Model & Weapon\Gun.PNG").convert_alpha()
        self.gun_offset_y = 100
        self.gun_rect = self.gun_image.get_rect(midbottom=(self.rect.centerx, self.rect.centery + self.gun_offset_y))
        self.jump_velocity = 0  # Initialize jump velocity
        self.gravity = 1
        self.jumpsound = SoundManager()

# Inside the Player class update method
    def update(self, keys):
        if keys[pygame.K_LEFT]:
            self.rect.x -= 5
        if keys[pygame.K_RIGHT]:
            self.rect.x += 5

        if keys[pygame.K_SPACE] and self.rect.bottom >= 640:
            self.jumpsound.jump()
            self.jump_velocity = -15

        self.rect.y += self.jump_velocity
        self.jump_velocity += self.gravity
        
        if self.rect.bottom >= 640:
            self.rect.bottom = 640
            self.jump_velocity = 0
            if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
                self.current_frame = (self.current_frame + 1) % (5 * 6)
                self.character_image = self.run_frames[self.current_frame // 5 % 6]
            else:
                self.character_image = self.run_frames[0]

        self.image = self.character_image
        self.gun_rect.midbottom = (self.rect.centerx, self.rect.centery + self.gun_offset_y)


class SpawnAnimation(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.spawn_images = [pygame.image.load(fr'D:\Python Mini Projects\ZombieZapper\Game_Assets\2.Enemies\EnemySpawn\Spawn{i}.PNG').convert_alpha() for i in range(1, 8)]
        self.image = self.spawn_images[0]
        self.rect = self.image.get_rect(midbottom=(x, y))
        self.animation_count = 0
        self.animation_speed = 15 
        self.current_frame = 0

    def update(self):
        self.animation_count += self.animation_speed
        current_frame = (self.animation_count // 60) % len(self.spawn_images)
        self.image = self.spawn_images[current_frame]
        if current_frame == len(self.spawn_images) - 1:
            self.kill()


class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_images = [pygame.image.load(fr'D:\Python Mini Projects\ZombieZapper\Game_Assets\2.Enemies\Enemy2\Enemy2_{i}.PNG').convert_alpha() for i in range(1, 7)]
        self.images = self.original_images.copy()
        self.image = self.images[0]
        self.rect = self.image.get_rect(midbottom=(x, y))
        self.animation_count = 0
        self.animation_speed = random.randint(8, 18)   
        self.current_frame = 0
    def update(self, enemy_speed):
        self.rect.x -= enemy_speed

        if self.rect.right <= 0:
            self.kill()
        self.images = [pygame.transform.flip(img, True, False) for img in self.original_images]

        self.animation_count += self.animation_speed
        current_frame = (self.animation_count // 60) % len(self.images)
        self.image = self.images[current_frame]


class AnotherEnemy(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_images = [pygame.image.load(fr'D:\Python Mini Projects\ZombieZapper\Game_Assets\2.Enemies\Enemy1\Enemy1_{i}.PNG').convert_alpha() for i in range(1, 7)]
        self.images = self.original_images.copy()
        self.image = self.images[0]
        self.rect = self.image.get_rect(midbottom=(x, y))
        self.animation_count = 0
        self.animation_speed = random.randint(8, 15) 
        self.current_frame = 0
    def update(self, enemy_speed):
        self.rect.x -= enemy_speed

        if self.rect.right <= 0:
            self.kill()

        self.images = [pygame.transform.flip(img, True, False) for img in self.original_images]

        self.animation_count += self.animation_speed
        current_frame = (self.animation_count // 60) % len(self.images)
        self.image = self.images[current_frame]


class BulletAnimation:
    def __init__(self) -> None:
        # Bullet Settings
        self.bullet_image = pygame.image.load(r"D:\Python Mini Projects\ZombieZapper\Game_Assets\1.MainCharacterArt\Model & Weapon\Bullet.PNG").convert_alpha()
        self.bullet_speed = 20
        self.bullets = []
        self.bullet_rect = self.bullet_image.get_rect(center=(400, 800))
        self.sound = SoundManager()

    def update(self, all_enemies_group, spawn_animations):
        # Bullet Movement and Collision Check
        new_bullets = []

        for bullet in self.bullets:
            bullet.x += self.bullet_speed

            # Check if bullet is out of screen
            if bullet.right < 0:
                bullet.kill()  # Remove the bullet if it is out of screen
                continue  # Skip to the next iteration

            # Check for bullet-enemy collisions
            hit_enemy = False
            for enemy in all_enemies_group:
                if bullet.colliderect(enemy.rect):
                    # Play spawn animation sound
                    self.sound.play_spawn_animation_sound()

                    # Create a SpawnAnimation at the enemy's position
                    spawn_animation = SpawnAnimation(enemy.rect.midbottom[0], enemy.rect.midbottom[1])
                    spawn_animations.add(spawn_animation)

                    # Remove the enemy
                    all_enemies_group.remove(enemy)
                    hit_enemy = True

                    break  # Exit the enemy loop if a collision occurred

            # If the bullet didn't hit an enemy, add it to the new list
            if not hit_enemy:
                new_bullets.append(bullet)

        # Update the bullets list with the new one
        self.bullets = new_bullets
        
    def fire(self, event, player):
        if event.key == pygame.K_f:
            self.sound.play_gun_shot_sound()
            self.new_bullet = self.bullet_rect
            self.bullet_offset = (50, 120)
            self.new_bullet.midbottom = (player.gun_rect.midtop[0] + self.bullet_offset[0], player.gun_rect.midtop[1] + self.bullet_offset[1])
            self.bullets.append(self.new_bullet)


class SoundManager:
    def __init__(self):
        pygame.mixer.init()

        # Load sounds
        self.main_sound = pygame.mixer.Sound(r"D:\Python Mini Projects\ZombieZapper\Game_Assets\5. Music\Enigma-Long-Version-Complete-Version(chosic.com).mp3")
        self.gun_shot_sound = pygame.mixer.Sound(r"D:\Python Mini Projects\ZombieZapper\Game_Assets\6. Sound Affects\Gunshot.wav")
        self.spawn_animation_sound = pygame.mixer.Sound(r"D:\Python Mini Projects\ZombieZapper\Game_Assets\6. Sound Affects\EnemySpawn.wav")
        self.jump_sound = pygame.mixer.Sound(r"D:\Python Mini Projects\ZombieZapper\Game_Assets\6. Sound Affects\Jump.wav")
        
    def sound_setting(self):
        # Play main sound in a loop
        self.main_sound.play(loops=-1)
        self.main_sound.set_volume(0.6)
        
    def play_gun_shot_sound(self):
        self.gun_shot_sound.set_volume(0.3)
        self.gun_shot_sound.play()

    def play_spawn_animation_sound(self):
        self.spawn_animation_sound.set_volume(0.5)
        self.spawn_animation_sound.play()

    def jump(self):
        self.jump_sound.play()
    
    def stop_main_sound(self):
        pygame.mixer.stop()

    def set_main_sound_volume(self, volume):
        self.main_sound.set_volume(volume)