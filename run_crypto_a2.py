from crypto_env_a2 import Crypto
from DQN_modified_a2 import DeepQNetwork
import numpy as np

initial_cash = 100000

env = Crypto(name='BTC-USD', data_path='./test.csv', start_cash=initial_cash, fee=0.001, drawdown_call=0.0001, fixed_stake=0.01, period=240)

RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      e_greedy_increment=0.001,
                      replace_target_iter=300,
                      memory_size=30000,
                      output_graph=False
                      )
total_steps = 0
total_length = env.length
#total_length = 300000
profit = []
#profit = 0
for i_episode in range(total_length):

    observation = env.reset()

    ep_r = 0
    while True:


        RL.conv_keep_prob = 1.0
        RL.gru_keep_prob = 1.0
        RL.dense_keep_prob = 1.0
        action = RL.choose_action(observation)

        observation_, reward, done = env.step(action)

        # the smaller theta and closer to center the better

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 300:

            RL.conv_keep_prob = 0.5
            RL.gru_keep_prob = 0.5
            RL.dense_keep_prob = 0.5
            RL.learn()
        #if total_steps % 300 == 0:
            #print(str(round((100.0*(total_steps+1))/total_length,2)) + '%')
            #print(local_steps,env.cash, env.portfolio, env.amt, env.value, env.start_value)

        #if local_steps >= 40000:
        #    done = True

        if done:
            profit.append(env.portfolio - initial_cash)
            #profit += (env.portfolio - initial_cash)
            print('episode: %d/%d'%(i_episode,total_length),
                  'ep_r: ', round(ep_r, 4),
                  'portfolio: ', round(env.portfolio,6),
                  'epsilon: ', round(RL.epsilon,6),
                  #'learning_rate: ', RL.lr2,
                  'avg profit: ', round(np.mean(profit),6),
                  'amt: ', round(env.amt,8),
                  'value: ', round(env.next_value,8))
            break

        observation = observation_
        total_steps += 1
        #local_steps += 1
    #if i_episode >= 100 and i_episode % 100:
    #    RL.save()

RL.plot_cost()
