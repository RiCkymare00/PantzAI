import PantzAI

for i in range(1,100):
    max_timesteps = i
    episodes = 1
    
    print('timesteps: ',max_timesteps)
    PantzAI.HH_ansatzRL(max_timesteps,episodes)
print('End of the simulation')

