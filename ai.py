import torch
import torch.nn as nn
import torch.optim as optim
import collections
import numpy as np
import random
import math
import poke_env
from poke_env.player.player import Player
from poke_env.environment.env import _EnvPlayer,PokeEnv
from poke_env.battle import Battle,AbstractBattle
from poke_env.data import GenData
from collections import namedtuple

import gymnasium as gym
from gymnasium.spaces import Box,Discrete

import os
import dotenv

dotenv.load_dotenv()

EPS = 0.9
EPS_MIN = 0.04
DECAY = 10000
LR = 0.0001
DISCOUNT= 0.9

ACCOUNT = poke_env.AccountConfiguration(os.environ['PKMNUSERNAME1'], os.environ['PASSWORD1'])


class Memory(object):# Agent memory class
    def __init__(self,capacity):
        self.m = collections.deque([],maxlen= capacity)

    def sample(self,size):
        return random.sample(self.m,size)
    
    def push(self, *args):
        self.m.append(*args)
        

class DQN(nn.Module):# neutal net class 
    def __init__(self, observaitons, num_Of_Actions ):
        super().__init__()
        self.n1 = nn.Linear(observaitons,64)
        self.n2 = nn.Linear(64,64)
        self.n3 = nn.Linear(64,num_Of_Actions)
        
    def forward(self,x):
        x = nn.functional.relu(self.n1(x))
        x = nn.functional.relu(self.n2(x))
        x = nn.functional.relu(self.n3(x))
        return x


class PokeAI(Player): 
    def __init__(self,battleOpponent = None,account=ACCOUNT,replays = False):
        super().__init__(
            account_configuration= account,
            server_configuration=poke_env.LocalhostServerConfiguration,# change if using a custom server
            # opponent=battleOpponent,
            # start_challenging=True,
            battle_format="gen9randombattle",
            log_level=0
            # ,save_replays = "battles"
            )
        self.action_space = Discrete(13)
        self.steps = 0
        self.last_battle = None
        self.last_state = None
        self.qn = DQN(torch.tensor(len(self.describe_embedding().sample())),self.action_space.n).to(torch.device('cpu'))
        self.optimizer = optim.Adam(self.qn.parameters(),lr=LR)
        self.mem = Memory(1000)
        self.loss = []
        self.rewards = {}
        
    def choose_move(self, battle:Battle):

        #get state
        state = self.embed_battle(battle)
        state = torch.tensor(state,dtype=torch.float32)
        
        # print(f'State: {state}')

        # choose move based on epsilon greedy policy
        sample = random.random()
        eps = EPS* math.exp(-1 * self.steps / DECAY) 

        if eps < EPS_MIN: 
            eps = EPS_MIN
        
        self.steps += 1 
        
        #uncomment if not training
        # eps = 0

        if sample > eps:
            action = torch.argmax((self.qn(state))).item()
        
        else:
            action = self.action_space.sample()
        #get chosen action
    
        # print(battle.valid_orders[0])
        # print(action)
        # print(battle.active_pokemon)
        # print(len(battle.valid_orders))
        # if battle.force_switch:
        #     print([i for i in battle.team.values() if not i.fainted])
        
        if action not in range(len(battle.valid_orders)) and len(battle.valid_orders) != 0:
        #if not enough options? -> modulo the action by the number of availble choices
            
            action = action % len(battle.valid_orders)
            

        reward = self.calc_reward(battle)
        if self.last_state != None:
            self.mem.push((self.last_state,state,action,reward))

        self.last_state = state
        
        # print(battle.valid_orders)

        return battle.valid_orders[action]
        
    def calc_reward(self, battle:AbstractBattle):
        fainted_value = 0.0
        hp_value = 0.2
        number_of_pokemons = 6
        starting_value = 0.0
        status_value = 0.4
        victory_value = 2.5
        #reinventing a wheel because i cant get the wheels to fit
        if battle not in self.rewards.keys():
            self.rewards[battle] = starting_value
        
        current_value = 0.0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value
        
        self.rewards[battle] += current_value
        # print(f'current: {current_value}')
        # print(f'stored: {self.rewards.get(battle)}')
        return current_value 
        # return PokeEnv.reward_computing_helper(battle, fainted_value = 2.0,
        # hp_value = 0.2,
        # number_of_pokemons = 6,
        # starting_value = 0.0,
        # status_value = 0.4,
        # victory_value = 2.5)
    
    def embed_battle(self, battle:AbstractBattle):
        bp = []
        multiplier = []
        final = []
        moves = battle.available_moves
        
        for i,move in enumerate(moves):
            bp.append(move.base_power)

            multiplier.append(move.type.damage_multiplier(
                battle.opponent_active_pokemon.type_1,
                battle.opponent_active_pokemon.type_2,
                type_chart= GenData.from_gen(battle.gen).type_chart
            ))

        team = len(battle.team) - len([i for i in battle.team.values() if i.fainted]) 

        opp_team = len(battle.opponent_team) - len([i for i in battle.opponent_team.values() if i.fainted])

        canTera = battle.can_tera
        if canTera != None:
            canTera = 1 
        else:
            canTera = 0

        # opp_canTera = battle._opponent_can_terrastallize
        
        # if opp_canTera== True:
        #     opp_canTera = 1
        # else:
        #     opp_canTera = 0
    

        if bp == []:
            for i in range(4):
                bp.append(0)
        
        if len(bp) < 4:
            while len(bp) != 4:
                bp.append(0)
        
        if multiplier == []:
            for i in range(4):
                multiplier.append(0)
        
        if len(multiplier) < 4:
            while len(multiplier) != 4:
                multiplier.append(0)
        
        
        # base power
        for i in bp:
            final.append(i)    
        #remainig  mons
        final.append(team)
        final.append(opp_team)
        #health
        final.append(round(battle.active_pokemon.current_hp_fraction,2))
        final.append(round(battle.opponent_active_pokemon.current_hp_fraction,2))
        #type (0-20)
        final.append(battle.active_pokemon.type_1.value)
        if battle.active_pokemon.type_2 == None:
            final.append(0)
        else:
            final.append(battle.active_pokemon.type_2.value)
        
        final.append(battle.opponent_active_pokemon.type_1.value)
        if battle.opponent_active_pokemon.type_2 == None:
            final.append(0)
        else:
            final.append(battle.opponent_active_pokemon.type_2.value)
        #tera
        final.append(canTera)
        # final.append(opp_canTera)

        for i in multiplier:
            final.append(i)
        
        # final.append("s")
        
        return final
    
    def describe_embedding(self):
        '''
        Box 1: base power(0-500)
        Box 2: remaining pokemon(0-6), current hp(0-1) (both sides) types(0-20)
        Box 3: Tera availabile(0-1) (both sides)
        Box 4: current damage multiplier(0-4)
        '''
        return gym.spaces.flatten_space(gym.spaces.Tuple(
            (Box(0,500,(4,),int),
            (Box(
                low = np.array([0,0,0,0,1,0,1,0]),
                high = np.array([6,6,1,1,20,20,20,20]),
                dtype =int)),
            (Box(0,1,(1,),dtype=int)),
            Box(0,4,(4,)))))
    
    def optimize(self, sample_size):
        #training function
        
        predictions = []
        average_loss = []

        if len(self.mem.m) < sample_size:
            return 
    
        obs = namedtuple("Observations",("state","next_state","action","reward"))
        sample = obs(*zip(*self.mem.sample(sample_size)))
        

        states,nexts,acts,rewards = sample

        l = 0 # itetator
        
        # get all the states from memory sample

        for s in states:
            
            
            pred = self.qn(torch.tensor(states[l],dtype=torch.float32)).to(torch.float32)
            
            predictions.append(pred)


            # calculates Q(s',a)
            next_pred = self.qn(torch.tensor(nexts[l],dtype=torch.float32)).to(torch.float32)
            next_val = torch.argmax(next_pred).item()
            
            new_pred = next_pred.clone()
            new_pred[next_val] = next_pred[next_val] + rewards[l]+ next_val * DISCOUNT
            new_pred.to(torch.float32)
            
            #calculates difference between action prediction and action taken
            self.optimizer.zero_grad()
            lossf = nn.MSELoss()
            loss = lossf.forward(pred,new_pred)
            loss.backward()       
            self.optimizer.step()
            average_loss.append(loss.detach())
            l += 1

        self.loss.append(sum(average_loss)/len(average_loss))
    
    def _battle_finished_callback(self, battle: AbstractBattle):
        self.optimize(32)
        print("Adapting...")

    


