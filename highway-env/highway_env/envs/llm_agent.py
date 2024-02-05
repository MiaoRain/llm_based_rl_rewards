###conda activate base
from .prompt_llm import *
from .merge_env import Scenario

# from prompt_llm import *
# from merge_env import Scenario
# # Group 19 Final Project: LLM-RL-AD-Agent
# GROUP MEMBERS:
# *   Ziqi Zhou zq.zhou@mail.uoft.ca 1009356224
# *   Jingyuan Zhang jymike.zhang@mail.utoronto.ca 1009146309
# *   Jingyue Zhang jingyuezjy.zhang@mail.utoronto.ca 1009268392
# ## Enviroment Set Up
# # Please upload the requirements.txt and MYKEY.txt file to /content/
# 
# # The first file contain requirements and dependencied that required for this project. MYKEY.txt should contain your API key.
# 
# # After the all rquired packages were downloaded ans installed, pleas **re-start** the session or use CTRL-M to clean the work space.
# 
# # The dependency files are in the same google drive.
# !pip install gym==0.22.0
#!pip install -r requirements.txt
# !pip install -r requirements.txt --ignore-installed requests
# !pip install --upgrade --force-reinstall pydantic
# !pip install openai==1.2.2
# !pip install numpy --upgrade
# !pip install tensorboardx gym pyvirtualdisplay
# !apt-get install -y xvfb ffmpeg
# ## Pre-Defined Prompt
# Basic system message, traffic rules and decision cautions.
# These rules can be modified to test if changing prompt leads to different behaviour pattern of our agent.

# ## TOOL Functions
# ### initialize openAI
from openai import OpenAI
# with open('MYKEY.txt', 'r') as f:
#     api_key = f.read()
# API_KEY = api_key
# ## Training model
# import highway_env
import numpy as np
import gymnasium as gym
from gym import spaces
import random
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3 import DQN,PPO

class LlmAgent():
    def __init__(self, road_info):
        # self.env = env
        # self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        self.sce = Scenario(road_info, vehicleCount=10 )
        self.frame = 0
        # self.obs = self.env.reset()
        self.done = False
        self.toolModels = [
            getAvailableActions(),
            getAvailableLanes(self.sce),
            getLaneInvolvedCar(self.sce),
            isChangeLaneConflictWithCar(self.sce),
            isAccelerationConflictWithCar(self.sce),
            isKeepSpeedConflictWithCar(self.sce),
            isDecelerationSafe(self.sce),
        ]
        self.pre_prompt = PRE_DEF_PROMPT()
        self.ACTIONS_ALL= {
            0: 'LANE_LEFT',
            1: 'IDLE',
            2: 'LANE_RIGHT',
            3: 'FASTER',
            4: 'SLOWER'
        }
    def llm_controller_run(self, env, controlled_vehicles):
        #input- env, self.controlled_vehicles;
        #output- llm_actions

        llm_actions = []
        rewards = []
        for i, ege_veh in enumerate(controlled_vehicles):  #
            prompt_info = self.prompt_engineer(ege_veh, env.road, env)#prompt engineer
            print("prompt_info:", prompt_info)
            llm_action = self.send_to_chatgpt(prompt_info)
            print("llm_action:", llm_action)
            llm_actions.append(llm_action)
        # env.env_method('set_llm_suggested_action', llm_action)


        # compute overall reward
        #     reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
        #              + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
        #              + self.config["MERGING_LANE_COST"] * Merging_lane_cost \
        #              + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)
        return llm_actions
    def retrun_sce(self):
        return self.sce
    def combined_reward(w1, w2, llmReward, RLReward, Sigmoid: bool):
        # Adjust these weights to change the reward shapping, use Sigmoid(bool) param to apply sigmoid function
        temp = w1 * llmReward + w2 * RLReward
        if Sigmoid:
            # Apply a sigmoid function to keep the combined reward between 0 and 1 if necessory
            #reward = 1 / (1 + np.exp(-combined_reward)) #src
            reward = 1 / (1 + np.exp(-temp))
        else:
            reward = temp
        return reward

    # def step(self, action):
    #     obs, reward, done, info = self.env.step(action)
    #     self.frame += 1
    #     self.obs = obs
    #     self.done = done
    #     self.sce.updateVehicles(obs, self.frame)
    #     return obs, reward, done, info

    # def step1(self, action): #src
    #     # Step the wrapped environment and capture all returned values
    #     obs, dqn_reward, done, truncated, info = self.env.step(action)
    #     llm_reward = self.calculate_custom_reward(action)
    #     # Adjust these weights to change the reward shapping
    #     Reward = combined_reward(0.8,0.2,llm_reward,dqn_reward,False)
    #     return obs, Reward, done, truncated, info

    def render(self):
        self.env.render()
    def close(self):
        self.env.close()
    def get_action_id_from_name(self, action_name, actions_all):
        """
        Get the action ID corresponding to the given action name.

        Parameters:
        action_name (str): The name of the action (e.g., 'LANE_LEFT').
        actions_all (dict): Dictionary mapping IDs to action names.

        Returns:
        int: The ID corresponding to the action name, or -1 if not found.
        """
        for id, name in actions_all.items():
            if name == action_name:
                return id
        return -1  # Return -1 or any suitable value to indicate 'not found'

    def send_to_chatgpt(self,  current_scenario):
        client = OpenAI(api_key="sk-mfSpwdDUbFLA1Sq0nGldKy5IY9StuRr9cbcuVRQWcsxsV2Kl",
                        base_url="https://api.chatanywhere.tech/v1")  # base_url="https://api.chatanywhere.com.cn/v1"
        # client = OpenAI(api_key=API_KEY,
        #                 base_url="https://api.chatanywhere.tech/v1" ) #
        # action_id = int(last_action)  # Convert to integer
        message_prefix = self.pre_prompt.SYSTEM_MESSAGE_PREFIX
        traffic_rules = self.pre_prompt.get_traffic_rules()
        decision_cautions = self.pre_prompt.get_decision_cautions()
        # action_name = ACTIONS_ALL.get(action_id, "Unknown Action")
        # action_description = ACTIONS_DESCRIPTION.get(action_id, "No description available")

        prompt = (f"{message_prefix}"
                  f"You, the 'ego' car, are now driving on a highway. You have already driven for some seconds.\n"
                  "Here is the current scenario:\n"
                  f"{current_scenario}\n\n"
                  "There are several rules you need to follow when you drive on a highway:\n"
                  f"{traffic_rules}\n\n"
                  "Here are your attention points:\n"
                  f"{decision_cautions}\n\n"
                  "Once you make a final decision, output it in the following format:\n"
                  "```\n"
                  "Final Answer: \n"
                  "    \"decision\": {\"<ego car's decision, ONE of the available actions>\"},\n"
                  "```\n")
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-16k-0613",  # "gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": prompt},
            ]
        )
        llm_response = completion.choices[0].message
        decision_content = llm_response.content
        llm_suggested_action = self.extract_decision(decision_content)
        print(f"llm action: {llm_suggested_action}")

        llm_action_id = self.get_action_id_from_name(llm_suggested_action, self.ACTIONS_ALL)
        llm_action = np.array([llm_action_id])
        return llm_action
    def set_llm_suggested_action(self, action):
        self.llm_suggested_action = action
    def calculate_custom_reward(self, action):
        if action == self.llm_suggested_action:
            return 1  # Reward for matching action
        else:
            return 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs  # Make sure to return the observation

    def get_available_actions(self):
        """Get the list of available actions from the underlying Highway environment."""
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        else:
            raise NotImplementedError(
                "The method get_available_actions is not implemented in the underlying environment.")

    def extract_decision(self, response_content):
        try:
            start = response_content.find('"decision": {') + len('"decision": {')
            end = response_content.find('}', start)
            decision = response_content[start:end].strip('"')
            if "LANE_LEFT" in decision:
                decision = "LANE_LEFT"
            elif "LANE_RIGHT" in decision:
                decision = "LANE_RIGHT"
            elif "FASTER" in decision:
                decision = "FASTER"
            elif "SLOWER" in decision:
                decision = "SLOWER"
            elif "IDLE" in decision:
                decision = "IDLE"
            return decision
        except Exception as e:
            print(f"Error in extracting decision: {e}")
            return None
    def prompt_engineer(self,  ege_veh,  road, env):
        # self.sce.updateVehicles(obs, frame, i)
        # Observation translation
        msg0 = available_action(self.toolModels, ege_veh, road, env)
        availabel_lane, msg1 = get_available_lanes(self.toolModels, ege_veh, road, env)
        msg2, lane_cars_id = get_involved_cars(self.toolModels, ege_veh, road, env, availabel_lane)
        #lane_cars_id -- {'lane_0': {'leadingCar': None, 'rearingCar': IDMVehicle #224: [173.94198546   0.        ]}}
        #availabel_lane -- {'currentLaneID': 'lane_0', 'leftLane': '', 'rightLane': ''}

        #msg1_info = next(iter(msg1.values()))
        # lanes_info = extract_lanes_info(msg1_info) #{'current': 'lane_3', 'left': 'lane_2', 'right': None}

        # lane_car_ids = extract_lane_and_car_ids(lanes_info, msg2) #{'current_lane': {'car_id': 'veh1', 'lane_id': 'lane_3'}, 'left_lane': {'car_id': 'veh3', 'lane_id': 'lane_2'}, 'right_lane': {'car_id': None, 'lane_id': None}}
        if availabel_lane["leftLane"] != "" or availabel_lane["rightLane"] != "": #如果需要换道
            safety_assessment = assess_lane_change_safety(self.toolModels, lane_cars_id, availabel_lane, ege_veh) #{'left_lane_change_safe': True, 'right_lane_change_safe': True}
        else:
            safety_assessment = "There is no need to assess lane change safety."
        safety_msg = check_safety_in_current_lane(self.toolModels, lane_cars_id, availabel_lane, ege_veh ) #{'acceleration_conflict': 'acceleration may be conflict with `veh1`, which is unacceptable.', 'keep_speed_conflict': 'keep lane with current speed may be conflict with veh1, you need consider decelerate', 'deceleration_conflict': 'deceleration with current speed is safe with veh1'}
        prompt_info = format_training_info(msg0, msg1, msg2, availabel_lane, lane_cars_id, safety_assessment,
                                      safety_msg)
        return prompt_info



class MyHighwayEnv(gym.Env):
    def __init__(self, vehicleCount=10):
        super(MyHighwayEnv, self).__init__()
        # base setting
        self.vehicleCount = vehicleCount
        # environment setting
        self.config = {
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False,
                "vehicles_count": vehicleCount,
                "see_behind": True,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": np.linspace(0, 32, 9),
            },
            "duration": 40,
            "vehicles_density": 2,
            "show_trajectories": True,
            "render_agent": True,
        }
        self.env = gym.make("highway-v0")
        self.env.configure(self.config)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,high=np.inf,shape=(10,5),dtype=np.float32
        )
    def combined_reward(self, w1,w2,llmReward,RLReward,Sigmoid:bool):
      # Adjust these weights to change the reward shapping, use Sigmoid(bool) param to apply sigmoid function
      temp = w1*llmReward + w2*RLReward
      if Sigmoid:
        # Apply a sigmoid function to keep the combined reward between 0 and 1 if necessory
        reward = 1 / (1 + np.exp(- self.combined_reward))
      else:
        reward = temp
      return reward
    def step(self, action): #src
        # Step the wrapped environment and capture all returned values
        obs, dqn_reward, done, truncated, info = self.env.step(action)
        llm_reward = self.calculate_custom_reward(action)
        # Adjust these weights to change the reward shapping
        Reward = self.combined_reward(0.8,0.2,llm_reward,dqn_reward,False)
        return obs, Reward, done, truncated, info

    def set_llm_suggested_action(self, action):
        self.llm_suggested_action = action
    def calculate_custom_reward(self, action):
        if action == self.llm_suggested_action:
            return 1  # Reward for matching action
        else:
            return 0

    def reset(self, **kwargs):
        self.obs = self.env.reset(**kwargs)
        return self.obs  # Make sure to return the observation

    def get_available_actions(self):
        """Get the list of available actions from the underlying Highway environment."""
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        else:
            raise NotImplementedError(
                "The method get_available_action")

    def init_env(self):
        env_demo = MyHighwayEnv(vehicleCount=10)
        # observation = env_demo.reset()
        # print("Initial Observation:", observation)
        # print("Observation space:", env_demo.observation_space)
        # Wrap the environment in a DummyVecEnv for SB3
        env_demo = DummyVecEnv([lambda: env_demo])  # Add this line
        available_actions = env_demo.envs[0].get_available_actions()
        return env_demo




if __name__ == '__main__':
    #create env and init

    my_env = MyHighwayEnv()
    env = my_env.init_env()
    model = DQN(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        # tensorboard_log=log_dir
    )
    obs = env.reset() #[1, 10, 5]
    action, _ = model.predict(obs)
    print("Action:", action)
    #create llm and action
    frame = 0
    llm_agent = LlmAgent(env)
    sce = llm_agent.retrun_sce()
    prompt_info = llm_agent.prompt_engineer(sce, obs, frame)
    llm_action = llm_agent.send_to_chatgpt(action, prompt_info, sce)
    print("llm_action:", llm_action)
    obs, reward, done, info = env.step(llm_action)
    print("reward：", reward)
    print("done!")

    # obs = [[[  1.       181.46942    8.        25.         0.      ],  [  1.       193.06259    4.        22.981533   0.      ],  [  1.       203.66267    4.        21.298485   0.      ],  [  1.       213.73927    4.        22.296495   0.      ],  [  1.       224.07132    4.        23.387285   0.      ],  [  1.       233.5289     8.        21.915598   0.      ],  [  1.       243.07896    8.        22.961285   0.      ],  [  1.       252.91997    4.        21.21295    0.      ],  [  1.       264.22998   12.        23.771507   0.      ],  [  1.       274.93527    4.        23.720556   0.      ]]]
