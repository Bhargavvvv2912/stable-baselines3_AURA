import sys
import gym
import numpy as np
import torch
import stable_baselines3 

def smoke_test():
    print("--- Starting Stable Baselines 3 (v1.1.0) Smoke Test ---")
    print(f"--> Gym Version: {gym.__version__}")
    
    try:
        # 1. Create Environment
        env = gym.make("CartPole-v1")
        env.reset()
        
        # 2. Test API Shape (The Regression Check)
        print("--> Testing Gym API compatibility...")
        action = env.action_space.sample()
        
        # CRITICAL: This is where the experiment happens.
        # Gym 0.19 (Safe) returns 4 values.
        # Gym 0.26 (Pip Upgrade) returns 5 values.
        results = env.step(action)
        
        if len(results) == 4:
            print(f"--> SUCCESS: API returned 4 values. Compatible with SB3 v1.1.0.")
        else:
            # If Pip upgrades to Gym 0.26, this will happen
            print(f"--> FAIL: API returned {len(results)} values. Incompatible with legacy code.")
            raise ValueError("too many values to unpack (expected 4)")

        # 3. Quick Train Loop
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        
        print("--> Training PPO Agent...")
        # v1.1.0 specific syntax check
        vec_env = make_vec_env("CartPole-v1", n_envs=1)
        model = PPO("MlpPolicy", vec_env, verbose=1, device='cpu')
        model.learn(total_timesteps=100)
        
        print("--- SMOKE TEST PASSED ---")
        sys.exit(0)

    except Exception as e:
        print(f"--- SMOKE TEST FAILED: {e} ---")
        sys.exit(1)

if __name__ == "__main__":
    smoke_test()