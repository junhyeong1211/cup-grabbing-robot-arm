{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNf7V+PSYBin65LxFxUg3PP"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!pip install gymnasium pybullet\n",
        "!pip install gymnasium pybullet stable-baselines3[extra]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iy79vb_cjdT0",
        "outputId": "d0b9b1d0-05e0-4529-ba54-6f591c4fd5b2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: gymnasium in /usr/local/lib/python3.12/dist-packages (1.2.1)\n",
            "Requirement already satisfied: pybullet in /usr/local/lib/python3.12/dist-packages (3.2.7)\n",
            "Requirement already satisfied: numpy>=1.21.0 in /usr/local/lib/python3.12/dist-packages (from gymnasium) (2.0.2)\n",
            "Requirement already satisfied: cloudpickle>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from gymnasium) (3.1.1)\n",
            "Requirement already satisfied: typing-extensions>=4.3.0 in /usr/local/lib/python3.12/dist-packages (from gymnasium) (4.15.0)\n",
            "Requirement already satisfied: farama-notifications>=0.0.1 in /usr/local/lib/python3.12/dist-packages (from gymnasium) (0.0.4)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import gymnasium as gym\n",
        "from gymnasium import spaces\n",
        "import numpy as np\n",
        "import pybullet as p\n",
        "import pybullet_data\n",
        "import time\n",
        "from stable_baselines3 import PPO\n",
        "import matplotlib.pyplot as plt\n",
        "from IPython.display import clear_output"
      ],
      "metadata": {
        "id": "2lxggi_2EYnO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class CupGraspingEnv(gym.Env):\n",
        "    def __init__(self):\n",
        "        super(CupGraspingEnv, self).__init__()\n",
        "\n",
        "        self.client = p.connect(p.DIRECT)\n",
        "        p.setAdditionalSearchPath(pybullet_data.getDataPath())\n",
        "\n",
        "        action_bound = 0.1\n",
        "        self.action_space = spaces.Box(low=-action_bound, high=action_bound, shape=(3,), dtype=np.float32)\n",
        "\n",
        "        obs_dim = 3 + 3 + 7\n",
        "        obs_low = np.array([-np.inf] * obs_dim)\n",
        "        obs_high = np.array([np.inf] * obs_dim)\n",
        "        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)\n",
        "\n",
        "        self.robotId = None\n",
        "        self.cupId = None\n",
        "        self.end_effector_index = 6\n",
        "        self.distance_to_cup = 0\n",
        "        self.max_steps_per_episode = 500 #200번 부족해서 500번으로 늘림\n",
        "        self.step_counter = 0\n",
        "        print(\"맞춤형 Gym 환경이 생성되었습니다!\")\n",
        "\n",
        "    def reset(self, seed=None, options=None):\n",
        "        self.step_counter = 0\n",
        "        p.resetSimulation(physicsClientId=self.client)\n",
        "        p.setGravity(0, 0, -9.8)\n",
        "\n",
        "        p.loadURDF(\"plane.urdf\")\n",
        "        self.robotId = p.loadURDF(\"kuka_iiwa/model.urdf\", [0, 0, 0], useFixedBase=True)\n",
        "\n",
        "        rand_x = np.random.uniform(0.4, 0.6)\n",
        "        rand_y = np.random.uniform(0.1, 0.3)\n",
        "        cup_start_pos = [rand_x, rand_y, 0.05]\n",
        "\n",
        "        cup_visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=0.1, rgbaColor=[0.8, 0.2, 0.2, 1])\n",
        "        cup_collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.04, height=0.1)\n",
        "        self.cupId = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=cup_collision_shape, baseVisualShapeIndex=cup_visual_shape, basePosition=cup_start_pos)\n",
        "\n",
        "        observation = self._get_obs()\n",
        "        self.distance_to_cup = np.linalg.norm(self.gripper_pos - self.cup_pos)\n",
        "\n",
        "        info = {}\n",
        "        return observation, info\n",
        "\n",
        "    def step(self, action):\n",
        "        current_gripper_state = p.getLinkState(self.robotId, self.end_effector_index)\n",
        "        current_gripper_pos = np.array(current_gripper_state[0])\n",
        "        target_position = current_gripper_pos + action\n",
        "\n",
        "        target_joint_angles = p.calculateInverseKinematics(self.robotId, self.end_effector_index, target_position)\n",
        "\n",
        "        if target_joint_angles: # IK 해가 있을 경우에만 실행\n",
        "            p.setJointMotorControlArray(\n",
        "                bodyIndex=self.robotId,\n",
        "                jointIndices=range(p.getNumJoints(self.robotId)),\n",
        "                controlMode=p.POSITION_CONTROL,\n",
        "                targetPositions=target_joint_angles\n",
        "            )\n",
        "\n",
        "        p.stepSimulation()\n",
        "\n",
        "        observation = self._get_obs()\n",
        "        reward = self._compute_reward()\n",
        "        terminated = self._check_done()\n",
        "        info = {}\n",
        "        return observation, reward, terminated, False, info\n",
        "\n",
        "    def _get_obs(self):\n",
        "        gripper_state = p.getLinkState(self.robotId, self.end_effector_index)\n",
        "        self.gripper_pos = np.array(gripper_state[0])\n",
        "\n",
        "        cup_pos, _ = p.getBasePositionAndOrientation(self.cupId)\n",
        "        self.cup_pos = np.array(cup_pos)\n",
        "\n",
        "        joint_states = p.getJointStates(self.robotId, range(p.getNumJoints(self.robotId)))\n",
        "        joint_positions = [state[0] for state in joint_states]\n",
        "\n",
        "        observation = np.concatenate([self.gripper_pos, self.cup_pos, joint_positions])\n",
        "        return observation\n",
        "\n",
        "    def _compute_reward(self):\n",
        "        new_distance = np.linalg.norm(self.gripper_pos - self.cup_pos)\n",
        "        reward = (self.distance_to_cup - new_distance) * 100\n",
        "        self.distance_to_cup = new_distance\n",
        "\n",
        "        if new_distance < 0.05:\n",
        "            reward += 5000\n",
        "\n",
        "        return reward\n",
        "\n",
        "    def _check_done(self):\n",
        "        if self.distance_to_cup < 0.05:\n",
        "            print(\"성공! 컵에 도달했습니다.\")\n",
        "            return True\n",
        "\n",
        "        self.step_counter += 1\n",
        "        if self.step_counter > self.max_steps_per_episode:\n",
        "            return True\n",
        "\n",
        "        return False\n",
        "\n",
        "    def render(self):\n",
        "        view_matrix = p.computeViewMatrix(cameraEyePosition=[1,1,1], cameraTargetPosition=[0.5,0,0.5], cameraUpVector=[0,0,1])\n",
        "        proj_matrix = p.computeProjectionMatrixFOV(fov=60.0, aspect=1.0, nearVal=0.1, farVal=100.0)\n",
        "        (w, h, rgb, _, _) = p.getCameraImage(width=224, height=224, viewMatrix=view_matrix, projectionMatrix=proj_matrix)\n",
        "        return rgb\n",
        "\n",
        "    def close(self):\n",
        "        if self.client >= 0:\n",
        "            p.disconnect(self.client)\n",
        "            self.client = -1\n",
        "        print(\"PyBullet 시뮬레이션 연결이 종료되었습니다.\")\n"
      ],
      "metadata": {
        "id": "u_DbfCUuo7zt"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}