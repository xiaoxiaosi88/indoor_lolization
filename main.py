# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import traceback # 引入traceback用于打印详细错误
import tensorflow as tf # Import tensorflow for GPU check

# 导入自定义模块
from environment import SingleFloorLocalizationEnv
from agent import DQNAgent
from data_utils import load_and_process_data
# 导入配置
import config

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"\n--- Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs ---\n")
            print(tf.config.list_physical_devices()) #打印更多gpu信息
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print("Error setting memory growth:", e)
    else:
        print("\n--- No GPU found, using CPU. ---\n")

def main():
    print("Starting Indoor Localization DQN Training...")
    check_gpu() # Check and print GPU status

    # 从 config 文件加载参数
    N_APS = config.N_APS
    ACTION_DIM = config.ACTION_DIM
    HISTORY_LEN = config.HISTORY_LEN
    STATE_INPUT_DIM = config.STATE_INPUT_DIM

    TARGET_BUILDING_ID = config.TARGET_BUILDING_ID
    TARGET_FLOOR_ID = config.TARGET_FLOOR_ID
    RADGT = config.RADGT
    DELTA_IOU = config.DELTA_IOU
    MAX_STEPS_PER_EPISODE = config.MAX_STEPS_PER_EPISODE

    NUM_EPISODES = config.NUM_EPISODES
    TARGET_UPDATE_FREQ = config.TARGET_UPDATE_FREQ
    BUFFER_SIZE = config.BUFFER_SIZE
    BATCH_SIZE = config.BATCH_SIZE
    LEARNING_RATE = config.LEARNING_RATE
    GAMMA = config.GAMMA
    EPSILON_START = config.EPSILON_START
    EPSILON_DECAY_RATE = config.EPSILON_DECAY_RATE
    EPSILON_MIN = config.EPSILON_MIN

    MODEL_SAVE_FREQ = config.MODEL_SAVE_FREQ
    # 格式化文件名
    MODEL_FILENAME = config.MODEL_FILENAME_TEMPLATE.format(building=TARGET_BUILDING_ID, floor=TARGET_FLOOR_ID)
    PLOT_FILENAME = config.PLOT_FILENAME_TEMPLATE.format(building=TARGET_BUILDING_ID, floor=TARGET_FLOOR_ID)
    LOG_FILENAME = config.LOG_FILENAME_TEMPLATE.format(building=TARGET_BUILDING_ID, floor=TARGET_FLOOR_ID)
    DATA_CSV_FILENAME = config.DATA_CSV_FILENAME
    NO_SIGNAL_RSSI = config.NO_SIGNAL_RSSI
    MIN_DETECTED_RSSI = config.MIN_DETECTED_RSSI

    # --- 加载和处理数据 ---
    print(f"--- Loading data for B{TARGET_BUILDING_ID} F{TARGET_FLOOR_ID} ---")
    start_time = time.time()
    rssi_train, location_train, map_bounds_train, norm_params = load_and_process_data(
        filename=DATA_CSV_FILENAME,
        building_id=TARGET_BUILDING_ID,
        floor_id=TARGET_FLOOR_ID,
        radgt=RADGT,  # Pass radgt needed for norm estimation
        no_signal_value=NO_SIGNAL_RSSI,
        rssi_min_value=MIN_DETECTED_RSSI
    )

    if rssi_train is None:
        print("Exiting due to data loading error.")
        return

    num_aps_actual = rssi_train.shape[1]
    if num_aps_actual != N_APS:
        print(f"FATAL: Loaded data has {num_aps_actual} WAP columns, but config expects {N_APS}.")
        return
    # 确认状态维度正确
    calculated_state_dim = num_aps_actual + 3 + (ACTION_DIM * HISTORY_LEN)
    if calculated_state_dim != STATE_INPUT_DIM:
         print(f"Warning: Calculated STATE_INPUT_DIM ({calculated_state_dim}) mismatches config ({STATE_INPUT_DIM}). Using calculated value.")
         STATE_INPUT_DIM = calculated_state_dim # Use actual calculated dim

    num_training_samples = rssi_train.shape[0]
    print(f"Data loading & processing complete ({num_training_samples} samples) in {time.time() - start_time:.2f}s")

    # --- 初始化环境和智能体 ---
    print("Initializing Environment and Agent...")
    env = SingleFloorLocalizationEnv(
        radgt=RADGT, delta=DELTA_IOU, eta=config.ETA_REWARD, tau=config.TAU_REWARD,
        max_steps=MAX_STEPS_PER_EPISODE, history_len=HISTORY_LEN, action_dim=ACTION_DIM
    )

    agent = DQNAgent(
        state_input_dim=STATE_INPUT_DIM, action_dim=ACTION_DIM, history_len=HISTORY_LEN,
        norm_params=norm_params,
        learning_rate=LEARNING_RATE, gamma=GAMMA,
        epsilon=EPSILON_START, epsilon_decay=EPSILON_DECAY_RATE, epsilon_min=EPSILON_MIN,
        buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ
    )

    # --- 加载之前的模型 ---
    if os.path.exists(MODEL_FILENAME):
        agent.load(MODEL_FILENAME) # Agent's load method has print statements

    # --- 打印最终确认的配置 ---
    print(f"\n--- Starting Training ---")
    print(f" Building: {TARGET_BUILDING_ID}, Floor: {TARGET_FLOOR_ID}")
    print(f" Episodes: {NUM_EPISODES}, Max Steps/Ep: {MAX_STEPS_PER_EPISODE}")
    print(f" Data Samples: {num_training_samples}")
    print(f" State Input Dim: {STATE_INPUT_DIM} (APs:{num_aps_actual}+Win:3+Hist:{ACTION_DIM * HISTORY_LEN})")
    print(f" Action Dim: {ACTION_DIM}")
    print(f" Learning Rate: {LEARNING_RATE}, Gamma: {GAMMA}")
    print(f" Epsilon: {agent.epsilon:.4f} -> {EPSILON_MIN:.4f} (Decay Rate: {EPSILON_DECAY_RATE})")
    print(f" Buffer Size: {BUFFER_SIZE}, Batch Size: {BATCH_SIZE}")
    print(f" Target Update Freq (steps): {TARGET_UPDATE_FREQ}")
    print(f" Save Freq (episodes): {MODEL_SAVE_FREQ}")
    print(f" Model Filename: {MODEL_FILENAME}")
    print(f" Map Bounds: {map_bounds_train}")
    print("-" * 40)

    # --- 训练日志和统计 ---
    try:
        log_file = open(LOG_FILENAME, 'w')
        log_file.write("Episode,Steps,Reward,Success,Epsilon,AvgLoss,FinalIoW\n")
    except IOError as e:
        print(f"Error opening log file {LOG_FILENAME}: {e}. Exiting.")
        return

    episode_rewards = []
    average_losses = []
    success_flags = []
    step_counts = []
    final_iows = []

    # --- 训练循环 ---
    training_start_time = time.time()
    print("Beginning training loop...\n")
    for episode in range(NUM_EPISODES):
        ep_start_time = time.time()
        data_idx = episode % num_training_samples # Cycle through data
        state = env.reset(
            rssi_vector=rssi_train[data_idx],
            ground_truth_loc=tuple(location_train[data_idx]),
            bounds=map_bounds_train
        )

        if state is None:
             print(f"Warning: Episode {episode + 1} skipped due to env reset failure.")
             episode_rewards.append(-env.eta * MAX_STEPS_PER_EPISODE)
             success_flags.append(0)
             step_counts.append(0)
             final_iows.append(0.0)
             log_file.write(f"{episode + 1},0,{-env.eta * MAX_STEPS_PER_EPISODE:.2f},0,{agent.epsilon:.5f},NaN,NaN\n")
             log_file.flush()
             continue

        total_reward = 0.0
        done = False
        step = 0
        episode_loss_sum = 0.0
        num_losses_in_episode = 0
        is_success = 0
        last_iow = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            step += 1
            total_reward += reward
            last_iow = info.get('iow', 0.0)

            if next_state is None:
                print(f"Error: Env step returned None in Ep {episode+1}, Step {step}. Ending.")
                done = True # Force end episode
            else:
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                # Learn from experience
                loss = agent.replay()
                if loss is not None and not np.isnan(loss):
                    episode_loss_sum += loss
                    num_losses_in_episode += 1

            # Check success state directly from 'done' and 'reward'
            if reward == env.eta and done:
                is_success = 1

            # Safety break if steps exceed limit substantially
            if step >= MAX_STEPS_PER_EPISODE + 5:
                 print(f"Warning: Force breaking episode {episode+1} due to excessive steps.")
                 if not done: total_reward -= env.eta # Add penalty if not already terminated successfully/negatively
                 done = True


        # --- End of Episode Stats & Logging ---
        ep_duration = time.time() - ep_start_time
        avg_loss_ep = episode_loss_sum / num_losses_in_episode if num_losses_in_episode > 0 else np.nan
        episode_rewards.append(total_reward)
        success_flags.append(is_success)
        step_counts.append(step)
        if not np.isnan(avg_loss_ep): average_losses.append(avg_loss_ep)
        final_iows.append(last_iow)

        avg_loss_str = f"{avg_loss_ep:.5f}" if not np.isnan(avg_loss_ep) else "NaN"
        log_line = f"{episode+1},{step},{total_reward:.2f},{is_success},{agent.epsilon:.5f},{avg_loss_str},{last_iow:.4f}\n"
        try:
             log_file.write(log_line)
             log_file.flush()
        except Exception as log_e:
             print(f"Error writing to log file: {log_e}")


        # Print progress periodically
        if (episode + 1) % 100 == 0 or episode == NUM_EPISODES - 1:
            moving_avg_window = 100
            avg_rew = np.mean(episode_rewards[-moving_avg_window:]) if len(episode_rewards) >= moving_avg_window else np.mean(episode_rewards)
            avg_steps = np.mean(step_counts[-moving_avg_window:]) if len(step_counts) >= moving_avg_window else np.mean(step_counts)
            avg_succ = np.mean(success_flags[-moving_avg_window:])*100 if len(success_flags) >= moving_avg_window else np.mean(success_flags)*100 if success_flags else 0
            avg_loss_recent = np.nanmean(average_losses[-moving_avg_window:]) if len(average_losses) >= moving_avg_window else np.nanmean(average_losses)
            loss_recent_str = f"{avg_loss_recent:.5f}" if not np.isnan(avg_loss_recent) else "NaN"

            print(f"Ep:{episode+1:>6}/{NUM_EPISODES} | Steps:{avg_steps:5.1f} | MA Rew:{avg_rew:+7.2f} | "
                  f"MA Succ:{avg_succ:6.2f}% | Epsilon:{agent.epsilon:.4f} | MA Loss:{loss_recent_str} | Ep Time:{ep_duration:.2f}s")

        # Save model weights periodically
        if (episode + 1) % MODEL_SAVE_FREQ == 0:
            agent.save(MODEL_FILENAME)
            print(f"--- Model weights saved to {MODEL_FILENAME} at episode {episode+1} ---")

    # --- End of Training ---
    log_file.close()
    total_training_time = time.time() - training_start_time
    print("-" * 40)
    print(f"Training finished in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s.")
    agent.save(MODEL_FILENAME)
    print(f"Final model saved to {MODEL_FILENAME}")

    # --- Plotting Results ---
    try:
        if os.path.exists(LOG_FILENAME):
            log_data = pd.read_csv(LOG_FILENAME)
            plot_window_size = 200 # For smoothing plots

            fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

            # Reward Plot
            axs[0].plot(log_data['Episode'], log_data['Reward'], label='Episode Reward', alpha=0.3, linewidth=0.8)
            axs[0].plot(log_data['Episode'], log_data['Reward'].rolling(window=plot_window_size, min_periods=1).mean(),
                        label=f'Mov Avg Reward ({plot_window_size} eps)', color='red')
            axs[0].set_ylabel("Total Reward")
            axs[0].set_title(f"Training Rewards (B{TARGET_BUILDING_ID} F{TARGET_FLOOR_ID})")
            axs[0].legend(loc='upper left')
            axs[0].grid(True)

            # Success Rate Plot
            ax0_twin = axs[0].twinx()
            ax0_twin.plot(log_data['Episode'], log_data['Success'].rolling(window=plot_window_size, min_periods=1).mean() * 100,
                          label=f'Mov Avg Success ({plot_window_size} eps)', color='orange', linestyle='--')
            ax0_twin.set_ylabel("Success Rate (%)")
            ax0_twin.legend(loc='lower right')

            # Loss Plot
            valid_loss = log_data['AvgLoss'].dropna()
            if not valid_loss.empty:
                 axs[1].plot(valid_loss.index + 1, valid_loss, label='Avg Batch Loss', alpha=0.6, linewidth=0.8)
                 axs[1].plot(valid_loss.rolling(window=plot_window_size // 5, min_periods=1).mean().index + 1, valid_loss.rolling(window=plot_window_size // 5, min_periods=1).mean(),
                             label=f'Mov Avg Loss ({plot_window_size // 5} eps)', color='green')
                 axs[1].set_ylabel("Loss (MSE)")
                 axs[1].set_title("Average Training Loss")
                 axs[1].set_yscale('log')
                 axs[1].legend()
                 axs[1].grid(True)
            else:
                 axs[1].set_title("No valid loss data to plot.")

            # Epsilon Decay Plot
            axs[2].plot(log_data['Episode'], log_data['Epsilon'], label='Epsilon', color='purple')
            axs[2].set_ylabel("Epsilon Value")
            axs[2].set_title("Epsilon Decay Over Episodes")
            axs[2].legend()
            axs[2].grid(True)

            # Final IoW Plot
            axs[3].plot(log_data['Episode'], log_data['FinalIoW'], label='Final IoW per Episode', alpha=0.3, color='blue')
            axs[3].plot(log_data['Episode'], log_data['FinalIoW'].rolling(window=plot_window_size, min_periods=1).mean(),
                       label=f'Mov Avg Final IoW ({plot_window_size} eps)', color='cyan')
            axs[3].set_xlabel("Episode")
            axs[3].set_ylabel("IoW")
            axs[3].set_title("Final Intersection over Window (IoW) per Episode")
            axs[3].legend()
            axs[3].grid(True)


            plt.tight_layout()
            plt.savefig(PLOT_FILENAME)
            print(f"Training plot saved to {PLOT_FILENAME}")
            # plt.show() # Uncomment to display plot interactively

        else:
            print(f"Log file {LOG_FILENAME} not found. Cannot generate plots.")

    except Exception as plot_e:
         print(f"Error generating plots: {plot_e}")
         traceback.print_exc()

if __name__ == "__main__":
    main()