from crypto_env import Crypto
from DQN_modified import DeepQNetwork

env = Crypto(name='BTC-USD', data_path='./test.csv', start_cash=1000, fee=0.001, drawdown_call=10, fixed_stake=0.001, period=180)

RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      e_greedy_increment=0.00001,
                      replace_target_iter=6400,
                      memory_size=100000,
                      # output_graph=True
                      )
total_steps = 0
total_length = env.length

for i_episode in range(total_length):

    observation = env.reset()

    ep_r = 0
    while True:

        action = RL.choose_action(observation)

        observation_, reward, done = env.step(action)

        # the smaller theta and closer to center the better

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 300:
            RL.learn()
        #if total_steps % 300 == 0:
            #print(str(round((100.0*(total_steps+1))/total_length,2)) + '%')
            #print(local_steps,env.cash, env.portfolio, env.amt, env.value, env.start_value)

        #if local_steps >= 40000:
        #    done = True

        if done:
            print('episode: %d/%d'%(i_episode,total_length),
                  'ep_r: ', round(ep_r, 2),
                  'portfolio: ', env.portfolio,
                  ' epsilon: ', round(RL.epsilon,6))
            break

        observation = observation_
        total_steps += 1
        #local_steps += 1
    #if i_episode >= 100 and i_episode % 100:
    #    RL.save()

RL.plot_cost()