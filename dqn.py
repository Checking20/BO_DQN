import numpy as np
import keras


class DQN(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 memory_size=1000,
                 batch_size=32,
                 replace_loop=100,
                 e_greedy_increment=None,
                 show_info=True,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features

        self.lr = learning_rate
        self.gamma = reward_decay

        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = self.epsilon_max if e_greedy_increment is None else 0

        self.memory_counter = 0
        self.memory_size = memory_size
        self.memories = np.zeros(
            shape=(self.memory_size, self.n_features*2 + 2),
            dtype=np.float64,
        )

        self.batch_size = batch_size
        self.learning_step_counter = 0
        self.replace_loop = replace_loop
        self.optimizer = keras.optimizers.RMSprop(lr=self.lr)

        # evaluation network
        self.e_model = self._build_network()
        # target network
        self.t_model = self._build_network()

        self.show_info = show_info
        if self.show_info:
            print('learning_rate',self.lr)
            print('e_greedy', self.epsilon_max)

    def _build_network(self):
        hidden_cell_number = 10

        input_layer = keras.layers.Input(shape=(self.n_features,))

        hidden_layer = keras.layers.Dense(hidden_cell_number, use_bias=True, activation='relu')(input_layer)
        output_layer = keras.layers.Dense(self.n_actions, use_bias=True)(hidden_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mean_squared_error', optimizer=self.optimizer)

        return model

    # replace the parameters of target network with those of evaluation network
    def _parameters_replace(self):
        para = self.e_model.get_weights()
        self.t_model.set_weights(para)
        if self.show_info:
            print("parameters have been replaced!")

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        point = self.memory_counter % self.memory_size
        self.memories[point, :] = transition
        self.memory_counter += 1

    # choose action using e-greedy algorithm
    def choose_action(self, state):
        state = np.reshape(state, [-1]+list(state.shape))
        dice = np.random.uniform()
        if dice < self.epsilon:
            #  pick the action with the max Q value using target network
            best_actions = list(self.t_model.predict(state, verbose=0)[0])
            action = np.argmax(best_actions)
        else:
            # randomly pick a action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        if self.learning_step_counter % self.replace_loop == 0:
            self._parameters_replace()

        if self.memory_counter < self.batch_size or self.memory_counter < self.memory_size/2:
            # there aren't [enough] data for learning
            return

        draw_box = min(self.memory_counter, self.memory_size)
        selected_ids = np.random.choice(draw_box, size=self.batch_size)
        selected_memories = self.memories[selected_ids, :]

        states = selected_memories[:, :self.n_features]
        states_ = selected_memories[:, -self.n_features:]
        rewards = selected_memories[:, self.n_features + 1]
        action_idx = selected_memories[:, self.n_features].astype(int)

        e_values = self.e_model.predict(states, batch_size=self.batch_size, verbose=0)
        t_values = self.t_model.predict(states_, batch_size=self.batch_size, verbose=0)
        q_target = e_values.copy()

        batch_idx = np.arange(self.batch_size).astype(int)
        q_target[batch_idx, action_idx] = rewards + self.gamma * np.max(t_values, axis=1)

        self.e_model.fit(states, q_target, batch_size=self.batch_size, verbose=0)

        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increment

        self.learning_step_counter += 1
