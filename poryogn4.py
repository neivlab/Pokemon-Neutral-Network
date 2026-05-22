import torch
import asyncio
import poke_env
from poke_env.player.baselines import RandomPlayer
from ai import PokeAI
import matplotlib.pyplot as plt
import time

import os
import dotenv

dotenv.load_dotenv()

N_BATTLES = 10
x_battles= []

async def main():
    #player that picks random moves 
    opp = RandomPlayer(account_configuration=poke_env.AccountConfiguration(os.environ['OPPONENT'], os.environ['PASSWORD2']), battle_format="gen9randombattle")
    
    player = PokeAI(battleOpponent=opp,replays = False)
    #loads pre trained model data if exists
    player.qn.load_state_dict(torch.load('INSERT MODEL HERE'))
    turn = 0
    battles = 0
    last_state = None

    player.reset_battles()
    
    for battle in range(N_BATTLES):
        #resets values before each battle
        # player.reset()
        await player.battle_against(opp)
        # print(player.battles[battle])
        # for value in player.battles.values():
        #     print(value.observations)

    
    
        # while player.current_battle.finished != True:
        #     #each step
        #     print(f'\n||   TURN {turn}   || ')
        #     action = player.choose_move(player.current_battle)
        #     state,reward,done,_,_ = player.step(action)
            
        #     if player.last_state != None:
        #         player.mem.push((last_state,state,action,reward))
        #     last_state = state
            
        #     turn += 1

        # battles += 1
        # x_battles.append(battles)
        
        # player.optimize(32)
        
        #plots graph of average loss for each batch
        # plt.title("Average Loss")
        # plt.xlabel("battle")
        # plt.ylabel("loss")
        # plt.plot(x_battles,player.loss)
        # plt.ylim(bottom=0,top= max(player.loss) + 1)
        
    
    #save model
    torch.save(player.qn.state_dict(),'model_{}.pth'.format(int(time.time())))
    
    print(f'\n\n{player.username} won {player.n_won_battles} of {player.n_finished_battles} battles')
    print(f'\n{player.steps} actions')

    # file = open("results.txt", "a")
    # file.write(f'\n\nwon {player.n_won_battles} of {player.n_finished_battles} battles')
    # file.write(f'\n{turn} actions')
    # file.close()
    
    #saves graph of average loss for each batch
    # plt.savefig("loss_{}".format(int(time.time())))
    print("Done")

if __name__ == "__main__":
    # asyncio.get_event_loop().run_until_complete(main())
    asyncio.run(main())

