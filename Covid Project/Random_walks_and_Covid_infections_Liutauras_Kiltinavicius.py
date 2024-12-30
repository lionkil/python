#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:06:26 2022

@author: liutauras
"""


"""
Created on Sun Apr 10 10:58:56 2022

@author: liutauras
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import random
#Resolution of simulation is weird on small screens, works best with a monitor instead of laptop.
 #Infection model
def Pd(x):
    PD=(-18.19*np.log(x)+43.276)/100
    return PD

def P_I(Pd,t):
    B=2.8/100 
    q=0.238*60
    p=0.0563
    Ez=1
    Pi=1-np.exp(-Pd*(B*q*p*t)/(Ez*20/60))
    return Pi
#How infection occurs
def infection(player,ratio,frame,width,real_frame,total_frames):
    if player.infected==False:#So cant get infected twice
        if len(player.times_distances)!=0:#Very important to have this everything will fail without this because when len of the array is 0 cant access an index of the array
            total_change=(player.times_distances[-1][1]-player.times_distances[0][1])/ratio #ratio is the amount of frames per 1 min,this is total time spent within the infected range
            
            #Player1
            if total_change<=2:
                player.inf_prob=0
                
            elif total_change>2:
                average_distance=0
                
                
                for i in range(len(player.times_distances)):
                    average_distance+=np.sqrt(player.times_distances[i][2])#Adding the distance to each player
                    
                    
                average_distance=width*average_distance/(len(player.times_distances))  #Working out average social distance   
                player.inf_prob=P_I(Pd(average_distance),total_change)#Infection model
                x=random.uniform(0,1)#Using this against infection prob to see if infected or not
                if x<=player.inf_prob: #Infecting
                    player.color='orange'
                    player.infected=True #values i extract to plot graphs:
                    player.time_inf=np.append(player.time_inf,[player.id,frame/ratio,total_change,average_distance,player.inf_prob,player.infected,int(real_frame/total_frames)+1]) #Adding time infected and total exposure time
                    player.time_inf=np.array(np.array_split(player.time_inf,int(len(player.time_inf)/6)))
                    player.total_time+=total_change

  
                elif x>=player.inf_prob:

                    player.total_time+=total_change
                    player.time_inf=np.append(player.time_inf,[player.id,frame/ratio,player.total_time,average_distance,player.inf_prob,player.infected,int(real_frame/total_frames)+1])
                    player.time_inf=np.array(np.array_split(player.time_inf,int(len(player.time_inf)/6)))
    
                #print(player.id,player.inf_prob,total_change,average_distance)
                player.times_distances=[]
                average_distance=0
            
        
    return player
    

class Player():#Assigning a player with the specified attributes
    def __init__(self,id=0,r=np.zeros(2),v=np.zeros(2),R=1e-2,color='black',inf_prob=0,times_distances=[],time_inf=[],infected=False,total_time=0):
        self.id, self.r,self.v,self.R,self.color,self.inf_prob,self.times_distances,self.time_inf,self.infected,self.total_time=id,r,v,R,color,inf_prob,times_distances,time_inf,infected,total_time

class Simulation():
    #Random bounds
    X=1
    Y=1
    frames=0
    N=30 #number of ppl
    width=20 #width of square of party
    ratio=6 #frames per 1min
    total_frames=1080
    inf_rad=10
    real_frame=0
    
    def __init__(self,dt=1e-2,N=30):
        self.dt,self.N=dt,N
        self.players=[Player(i) for i in range(self.N)] #Initialising 30 players
    
        
    def susceptibility(self): # checking distances between two players thats why two loops
        for player1 in self.players:
            for player2 in self.players:
                if player1.id==player2.id:#Makes sure infected cant infect itself
                    continue
                r1,r2=player1.r,player2.r
                if player1.color=='red' or player2.color=='red':
                    
                    if np.dot(r1-r2,r1-r2) <=(self.inf_rad/self.width)**2: #checking if within a 10m radius
      
                         #These if statements will add the id,frame and distance^2 from infected particle to individual arrays   
                        if player1.color=='red':

                            player2.times_distances=np.append(player2.times_distances,[player2.id,self.frames,np.dot(r1-r2,r1-r2)])
                            player2.times_distances=np.array(np.array_split(player2.times_distances, int(len(player2.times_distances)/3)))
    
                        elif player2.color=='red':
                            player1.times_distances=np.append(player1.times_distances,[player1.id,self.frames,np.dot(r1-r2,r1-r2)])
                            player1.times_distances=np.array(np.array_split(player1.times_distances, int(len(player1.times_distances)/3)))
                 
                    
                    
                        if self.frames==self.total_frames:#Getting infected if inside circle on last frame
                            
                            #Player1
                            infection(player1,self.ratio,self.frames,self.width,self.real_frame,self.total_frames)
                            #Player2
                            infection(player2,self.ratio,self.frames,self.width,self.real_frame,self.total_frames)
  
                        #If player spends any time in the susceptible radius of infected and leaves:  
                    
                    elif (len(player2.times_distances)!=0 or len(player1.times_distances)!=0) and (np.dot(r1-r2,r1-r2) <=(self.inf_rad/self.width)**2+0.01):

                        #Player1
                        infection(player1,self.ratio,self.frames,self.width,self.real_frame,self.total_frames)
                        #Player2
                        infection(player2,self.ratio,self.frames,self.width,self.real_frame,self.total_frames)
   

                        
  
    def collisions(self):
        
        for player1 in self.players:
            
            x,y=player1.r
            if ((x>self.X/2-player1.R) or (x <-self.X/2+player1.R)):
                player1.v[0] *=-1#reversing velocity if hits boundary
                
            if ((y>self.Y/2-player1.R) or (y <-self.Y/2+player1.R)):
                player1.v[1] *=-1
            
        
                
    timer=time.time()
    i=0
    def gathering(self):#The gathering ofg players
        time_intervals=np.arange(0,100000,10)
        if (time.time()-self.timer)>time_intervals[2*self.i+1]+1:
            self.i+=1
    
    
        for player1 in self.players:
            for player2 in self.players:
                if player1.id==player2.id:
                    continue
                if time_intervals[2*self.i]<(time.time()-self.timer)<time_intervals[2*self.i+1]:
                    r1,r2=player1.r,player2.r
                    if np.dot(r1-r2,r1-r2) <=0.05**2:
                        
                        if np.random.randint(1,1000)<=10: #Chance that they will stop
                            player1.v=np.zeros(2)
                            player2.v=np.zeros(2)
                            
                if time_intervals[2*self.i+1]<(time.time()-self.timer)<time_intervals[2*self.i+2]:
                    if player1.v[0]==0 and player1.v[1]==0:
                        player1.v=sim.X/2*np.array([random.uniform(-1,1),random.uniform(-1,1)])
                        
                    if player2.v[0]==0 and player2.v[1]==0:
                        player1.v=sim.X/2*np.array([random.uniform(-1,1),random.uniform(-1,1)])
                    
                            
        
                                
                
    
       

    def increment(self):
        self.frames+=1
        self.real_frame+=1
        
        if self.frames>1080: #Repeating the simulation 1080frames=3 hours
            self.frames=0
            for player in sim.players:
                player.r=np.random.uniform([-sim.X/2+player.R,-sim.Y/2+player.R], #I added the player.R so doesnt get stuck on the border during collisions incase its first 
                               [sim.X/2-+player.R,sim.Y/2-+player.R],size=2)#position is on the border
    
                player.v=sim.X/2*np.array([random.uniform(-1,1),random.uniform(-1,1)])
                if player.color=='orange':
                    player.color='black'
                player.inf_prob=0
                player.time_inf=[]
                player.total_time=0
                player.infected=False
                player.times_distances=[]
                
            
            self.players[0].color='red'
            self.players[0].infected=True

            
            
        self.collisions()
        self.gathering()
        self.susceptibility()
        
        for player in self.players:
            player.r+= self.dt *player.v #Animating the movement
        #Writing to a file so i can run the simulation for lots of iteration
        if sim.frames==sim.total_frames: #Will write to file every 1080 frames
            print(int(sim.real_frame/sim.total_frames))
            times=[]
            for player in self.players:
                if len(player.time_inf)!=0:
                    for i in range(len(player.time_inf)):
                        times.append(player.time_inf[i])
     
            times=np.array(times)
            for i in range(len(times)): #To make it easier to read from file
                times[i]=[format(x,'f') for x in times[i]]

            
            f=open('Covid infections p=0.0563.txt',mode='a')
            for i in range(len(times)):
                for item in times[i]:
                    
                    f.write("%s " %item)
                f.write("\n")

            f.close()
        
                
                
    def player_pos(self):
        positions=[player.r for player in self.players]
        return positions
    
    def player_colors(self):
        colors=[player.color for player in self.players]
        return colors
    

    



sim=Simulation()

#Making infected player red
sim.players[0].color='red'
sim.players[0].infected=True
#Assigning initial velocities and positions
for player in sim.players:
    player.r=np.random.uniform([-sim.X/2+player.R,-sim.Y/2+player.R], #I added the player.R so doesnt get stuck on the border during collisions incase its first 
                               [sim.X/2-+player.R,sim.Y/2-+player.R],size=2)#position is on the border
    
    player.v=sim.X/2*np.array([random.uniform(-1,1),random.uniform(-1,1)])

    
#Plotting code
fig,ax=plt.subplots()
ax.set_xticks([]),ax.set_yticks([])#getting rid of axis'
ax.set_title('Covid-19 Simulation')
scatter = ax.scatter([],[],s=10)


def init():
    ax.set_xlim(xmin=-sim.X/2,xmax=sim.X/2)#seting limits
    ax.set_ylim(ymin=-sim.Y/2,ymax=sim.Y/2)

    return scatter,

def update(frame):
    sim.increment()
    pos=np.array(sim.player_pos())#asiggning initial positions
    scatter.set_offsets(pos)
    scatter.set_color(sim.player_colors()) #asigning initial colours
    return scatter,
#the key animation to this whole thing
anim=FuncAnimation(fig,update,frames=range(325000),init_func=init,blit=True,interval=1/30, repeat=False)






