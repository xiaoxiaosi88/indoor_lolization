# agent.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import random
from collections import deque
import os
import copy # For deepcopy if needed inside remember (less crucial now)

class DQNAgent:
    """ DQN Agent including normalization logic. """
    def __init__(self, state_input_dim, action_dim, history_len, norm_params,
                 learning_rate=0.0005, gamma=0.1,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=64, target_update_freq=100):

        if state_input_dim <= 0: raise ValueError("state_input_dim must be positive.")
        self.state_input_dim = int(state_input_dim)
        self.action_dim = int(action_dim)
        self.history_len = int(history_len)
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.initial_epsilon = float(epsilon)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.learning_rate = float(learning_rate)
        self.target_update_freq = int(target_update_freq)
        self.learn_step_counter = 0

        # --- Normalization parameters handling ---
        self.norm_params = norm_params
        self.use_normalization = False
        if norm_params and isinstance(norm_params, dict):
            # Load params, ensuring float type and providing defaults
            self.rssi_mean = float(norm_params.get('rssi_mean', 0.0))
            self.rssi_std = float(norm_params.get('rssi_std', 1.0))
            if abs(self.rssi_std) < 1e-6: self.rssi_std = 1.0 # Prevent division by zero/very small numbers

            self.coords_mean = np.array(norm_params.get('coords_mean', [0.0, 0.0]), dtype=np.float32)
            self.coords_std = np.array(norm_params.get('coords_std', [1.0, 1.0]), dtype=np.float32)
            self.coords_std[np.abs(self.coords_std) < 1e-6] = 1.0 # Prevent zero std dev

            self.radius_mean = float(norm_params.get('radius_mean', 0.0))
            self.radius_std = float(norm_params.get('radius_std', 1.0))
            if abs(self.radius_std) < 1e-6: self.radius_std = 1.0

            self.use_normalization = True
            print("Agent: Normalization parameters loaded and enabled.")
            print(f"  RSSI: mean={self.rssi_mean:.2f}, std={self.rssi_std:.2f}")
            print(f"  Coords: mean={self.coords_mean}, std={self.coords_std}")
            print(f"  Radius: mean={self.radius_mean:.2f}, std={self.radius_std:.2f}")
        else:
            print("Warning: Normalization params not provided or invalid. Normalization disabled.")
        # -----------------------------------------

        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model() # Initial weight copy
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = losses.MeanSquaredError()

    def _build_model(self):
        """Builds the Neural Network Model."""
        model = models.Sequential([
            layers.InputLayer(input_shape=(self.state_input_dim,)),
            layers.Dense(2000, activation='relu'),
            layers.Dense(1448, activation='relu'),
            layers.Dense(548, activation='relu'),
            layers.Dense(self.action_dim, activation='linear') # Output layer for Q-values
        ], name="DQN_Model")
        model.compile(optimizer=self.optimizer, loss=self.loss_function) # Compile for easier saving/loading maybe
        return model

    def _prepare_state_input(self, state):
        """Converts state tuple to a standardized, flattened NumPy array."""
        if state is None: return np.zeros(self.state_input_dim, dtype=np.float32)

        rssi_vector, current_window, current_history = state
        # Validate window structure
        if current_window is None or not isinstance(current_window, (list, tuple)) or len(current_window) != 2 or not isinstance(current_window[0], (list, tuple)) or len(current_window[0]) != 2:
            print(f"Warning: Invalid window format in state: {current_window}. Returning zero vector.")
            return np.zeros(self.state_input_dim, dtype=np.float32)

        center_coords, radius = current_window
        # Ensure components are arrays and have correct types/shapes
        rssi_vector_np = np.array(rssi_vector, dtype=np.float32)
        window_vec = np.array([center_coords[0], center_coords[1], radius], dtype=np.float32)
        current_history_np = np.array(current_history, dtype=np.float32).reshape(-1) # Ensure flat

        # --- Apply Standardization ---
        if self.use_normalization:
            rssi_vector_np = (rssi_vector_np - self.rssi_mean) / self.rssi_std
            window_vec[0:2] = (window_vec[0:2] - self.coords_mean) / self.coords_std
            window_vec[2] = (window_vec[2] - self.radius_mean) / self.radius_std
        # --------------------------

        # Concatenate into the final flat state vector
        try:
            flat_state = np.concatenate([rssi_vector_np, window_vec, current_history_np])
        except ValueError as e:
            print(f"Error concatenating state parts:")
            print(f"  RSSI shape: {rssi_vector_np.shape}, dtype: {rssi_vector_np.dtype}")
            print(f"  Window shape: {window_vec.shape}, dtype: {window_vec.dtype}")
            print(f"  History shape: {current_history_np.shape}, dtype: {current_history_np.dtype}")
            raise e # Re-raise the error after printing info

        # Final dimension check
        if flat_state.shape[0] != self.state_input_dim:
            raise ValueError(f"Prepared state final dimension {flat_state.shape[0]} != expected {self.state_input_dim}")

        # Check for NaNs after normalization/concatenation
        if np.isnan(flat_state).any():
             print("Warning: NaN detected in prepared state vector.")
             # Option 1: Return zero vector (might hide issues)
             # return np.zeros(self.state_input_dim, dtype=np.float32)
             # Option 2: Raise error to stop training and debug
             raise ValueError("NaN detected in state vector after preparation.")

        return flat_state

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the buffer. Uses deepcopy for safety."""
        if state is None or next_state is None:
            # print("Warning: Skipping remembrance due to None state(s).")
            return
        # Deepcopy state components to prevent issues if original objects are mutated later
        # state_copy = copy.deepcopy(state)
        # next_state_copy = copy.deepcopy(next_state)
        # Note: Current env implementation returns new tuples/arrays, deepcopy less critical but safer
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        """Selects action using epsilon-greedy."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        prepared_state = self._prepare_state_input(state)
        state_tensor = tf.convert_to_tensor([prepared_state]) # Add batch dimension
        q_values = self.model.predict(state_tensor, verbose=0)[0] # Predict for single state
        return np.argmax(q_values)


    def replay(self):
        """Trains the main model using a batch from the replay buffer."""
        if len(self.memory) < self.batch_size: return None # Not enough samples

        minibatch = random.sample(self.memory, self.batch_size)

        # Prepare states and next states using _prepare_state_input
        current_states_np = np.array([self._prepare_state_input(s) for s, a, r, ns, d in minibatch], dtype=np.float32)
        next_states_np = np.array([self._prepare_state_input(ns) for s, a, r, ns, d in minibatch], dtype=np.float32)

        actions = np.array([a for s, a, r, ns, d in minibatch], dtype=np.int32)
        rewards = np.array([r for s, a, r, ns, d in minibatch], dtype=np.float32)
        dones = np.array([d for s, a, r, ns, d in minibatch], dtype=np.float32) # Bool to float

        # Predict future Q values with target network
        future_q_values_target = self.target_model.predict(next_states_np, batch_size=self.batch_size, verbose=0)
        max_future_q = np.max(future_q_values_target, axis=1)
        target_q_values = rewards + self.gamma * max_future_q * (1.0 - dones) # Q-target calculation

        # Train main model
        with tf.GradientTape() as tape:
            # Predict current Q values with main model
            q_values_main = self.model(current_states_np, training=True) # training=True if using Dropout/BN

            # Select Q-value for the action actually taken
            action_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), actions], axis=1)
            current_q_selected = tf.gather_nd(q_values_main, action_indices)

            # Calculate loss
            loss = self.loss_function(target_q_values, current_q_selected)

        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        valid_grads_vars = [(g, v) for g, v in zip(gradients, self.model.trainable_variables) if g is not None]
        if len(valid_grads_vars) < len(self.model.trainable_variables):
             print("Warning: Found None gradients during backprop. Optimizer step may be incomplete.")
        if not valid_grads_vars:
             print("Warning: No valid gradients found. Skipping optimizer step.")
             return loss.numpy()

        # Clip gradients for stability (optional)
        # clipped_gradients = [(tf.clip_by_norm(g, 1.0), v) for g, v in valid_grads_vars]
        # self.optimizer.apply_gradients(clipped_gradients)

        self.optimizer.apply_gradients(valid_grads_vars)


        # Update counters and target model
        self.learn_step_counter += 1
        self._update_epsilon()
        if self.learn_step_counter % self.target_update_freq == 0:
            self._update_target_model()

        return loss.numpy()

    def _update_target_model(self):
        """Copies weights from main model to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def _update_epsilon(self):
        """Decays epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load(self, name):
        """Loads model weights from a file."""
        if os.path.exists(name):
            try:
                self.model.load_weights(name)
                self._update_target_model() # Sync target model
                print(f"Agent: Model weights loaded from {name}")
            except Exception as e:
                print(f"Agent Error: Could not load weights from {name}. Starting fresh. Error: {e}")
        else:
            print(f"Agent: Weight file {name} not found. Starting fresh.")

    def save(self, name):
        """Saves model weights to a file."""
        try:
            self.model.save_weights(name)
            # print(f"Agent: Model weights saved to {name}") # Reduce verbosity
        except Exception as e:
            print(f"Agent Error: Could not save weights to {name}. Error: {e}")