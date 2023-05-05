""" 
By:Ankit kumar

photo and video credits's
flaticon.com
soundbible.com


"""
# Importing Required Libraries
import pygame
import random
import math
from pygame.locals import *


#Game Initiates from here
pygame.init()
#Setting The game window
dis  = pygame.display.set_mode((1000,700))

#Setting The background image 
playerimg = pygame.image.load('player.png')

#Loading Background Image
pygame.mixer.music.load('background.mp3')

#playing the Music -1 means the music will play for infinete
pygame.mixer.music.play(-1)

#setting the background image
background = pygame.image.load('background.png')

# Now we will set the co-ordinates of our Player
"""
    We have to set our player icon at a place where it is most accurate
    As our window width is 1000 and height is 700
    the Y co-ordinate is constant only X-axis Changes ,while playing
    the initial postion of our player is (250,550)
    
"""
#Setting X axis
playerx = 250

#Setting Y axis
playery = 550



""" the change is very important as it helps our player to move,
    It simple words it is the speed of our player in right-left direction
    further playerx will chnage but y speed is always constant 

 """

playerx_change = 0

playery_change = 0

"""
   In our game we have created multiple enemeies
   To create multiple enemies we are iterating the values
   of enemy properties into 6 enemies
   
"""

#Change in enemy x co-ordinates
enemyx_change = []

#Change in enemy y co-ordinates
enemyy_change = []

#Initial position of enemy at x co-ordinates
enemyx = []

#Initial position of enemy at y co-ordinates
enemyy = []

#We have to load the same image for all enemies
enemyimg = []





#No of enemies we want is setted by this variable
no_of_enemies = 6

#Now we will iterate the properties of enemies in all 6 enemies
for a in range(no_of_enemies):
    enemyx_change.append(4)
    enemyy_change.append(40)
    enemyx.append(random.randint(0,930))
    enemyy.append(random.randint(0,500))
    enemyimg.append(pygame.image.load('enemy.png'))


#Belows are the properties of bullet fired by our space ship

#Loading the image of Bullet    
bulletimg = pygame.image.load('bullet.png')

"""
   Here we will set the speed of bullet when it is fired
   we Don't have to change the x speed as the bullet will not change
   It's X-asis,We just have to change the y-axis.in pygame the co-ordinate
   system is inverted means to Go up we have to decrease the y speed of bullet
   
"""

bulletx_change = 0

#later on the program we will substrate the value of bullet x by change
bullety_change = 20

#Here we are setting the x Co-ordinate of bullet to 0
#But later we will make it relative to playerx

bulletx = 0

bullety = 550

#we will also keep a record of The player score


score = 0


textx = 10
texty = 10

""".Font() funtion takes 2 parameter.First is the Type/style of text
   the second Parameter is the size of our text

"""
font = pygame.font.Font('freesansbold.ttf',32)

font1 = pygame.font.Font('freesansbold.ttf',64)

#This is the function to display score 
def shows_score(x,y):
    #.render() takes three parameter. First is text you want to show,Seocond is for Anti-aliasing and
    #Third is for Colour it is a rgp tuple
    
    s = font.render("score:" + str(score),True,(255,244,241))
    #blit function helps to draw the image on surface first 
    dis.blit(s,(x,y))
"""
   Bullet state is the state of our bullet ready means the bullet can be fired
   fire state means it is fired
"""   
bullet_state = "ready"

#This is were math module is used
"""
   If the distance between the enemy and the bullet is less than 30
   Then we will assume that we hitted the enemy
   We will find the distance between them using distance formula
   we studied in class 9th

"""

def iscollision(enemyx,enemyy,bulletx,bullety):
    
    distance = math.sqrt((math.pow(enemyx-bulletx,2))+(math.pow(enemyy-bullety,2)))

    if distance < 30 and bullet_state == "fire":
        return True
    else:
        return False
#This is the function that will be displayed when our ship is invaded
#and this will display Game over
    
def over():
    
    sd = font1.render("Game Over",True,(255,255,25))

    dis.blit(sd,(350,250))
    
def player(x,y):
    
    dis.blit(playerimg,(x,y))

def enemy(x,y,i):

    dis.blit(enemyimg[i],(x,y))

def bullet(x,y):
#We are making the bullet state global
    global bullet_state
    #whenever this function is called the state will become fire(By pressing space)
    bullet_state = "fire"
    
    dis.blit(bulletimg,(x+10,y+10))
#in While loop all the states will constantly reloading    
while True:
    dis.fill((0,0,0))
    
    dis.blit(background,(0,0))
    # we are iterating for the event.get() which contains all the events happening like pressing a mouse
    for event in pygame.event.get():
        
        if event.type == QUIT:
            pygame.quit()
        #.KEYDOWN is activated when we press any key on keyboard
    
        if event.type == pygame.KEYDOWN:
            ##.K_RIGHT is for right key
            if event.key == K_RIGHT:
                print("right key is pressed")
                playerx_change =  9

             ##.K_LEFT is for left key   
            if event.key == K_LEFT:
                print("left key is pressed")
                playerx_change = -9
            ##.K_SPACE is for space key
            if event.key == K_SPACE:
              if bullet_state == 'ready':
                #below is the code that will run when when bullet is fired  
                bullet_sound = pygame.mixer.Sound('bullet.wav')
                bullet_sound.play()
                """here is a main logic.the bullet postion is same as of our players
                   but when it is fired we don't want it to be at postion of our player
                   so we copy the players postion in bulletx and then passing it so that
                   it don't change it's postion when our player moves after firing the
                   bullet
                """   
                bulletx = playerx
                
                bullet(bulletx,bullety)
        #This condition is for when we remove our finger from key

        if event.type == pygame.KEYUP:
            
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                
                #When we remove our finger from key we want it speed on x-axis to be zero                 

                playerx_change =0 
                
    #this will continously check the player x xo-oridinates
    playerx = playerx+playerx_change
    
    """
       Below we change set the co-ordinates so that it will not go
       out of boundaries
    """
    #If it goes out of 0 the player position will be changed to 0
    
    if playerx<=0:
        
        playerx = 0

    elif playerx>=930:

        playerx = 930
    #Here we are constantly changing the position of enemies
    #If our enemies goes beyond 550 y-axis we will postion it somewhere where we can't see it 

    for a in range(no_of_enemies):
        if enemyy[a] >= 550:  
           for j in range(no_of_enemies):
              enemyx[j] = 20000
           over()
           break
        enemyx[a] = enemyx[a] + enemyx_change[a]
     # Here we are constantly changing the posyion of our enemies(both x and y axis
     
        if enemyx[a] <= 0:
           enemyx_change[a] = 6
           enemyy[a]+=enemyy_change[a]

        elif enemyx[a]>=930:
           enemyx_change[a] = -6
           enemyy[a]+=enemyy_change[a]

        collison  = iscollision(enemyx[a],enemyy[a],bulletx,bullety)  
  
        # If collison is true we will change the co-ordinates of bullet and also it's state
        # then score will also will increased to 1
        # Also we will constantly be showing enemies if they dies
        if collison:
            
           bullety = 550
           score +=1
           bullet_state = "ready"
           print(score)
           enemyx[a] = random.randint(0,930)
           enemyy[a] = random.randint(0,500)
        enemy(enemyx[a],enemyy[a],a)  
        
        # If bullet is at 0 y co-ordinate
        # we will change it's co-ordinate to 550
        if bullety <= 0 :
           bullety = 550
           bullet_state = "ready"

        
    if bullet_state is "fire":
        bullet(bulletx,bullety)
        bullety -= bullety_change
   

        
    player(playerx,playery)
    
    shows_score(textx,texty)
    
    pygame.display.update() #This will constantly update the screen/Game
