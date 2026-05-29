from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, Scene, SensorConfig
import os
import numpy as np

class HumanAgent(AbstractAgent):
    """Privileged agent interface of human operator."""

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes the human agent object.
        :param trajectory_sampling: trajectory sampling specification
        """
        self._trajectory_sampling = trajectory_sampling

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self, cfg) -> None:
        """Inherited, see superclass."""
        self.pred_dir = os.path.join(cfg.pred_dir, cfg.split)

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene, token) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        raw_dir = os.path.join(self.pred_dir, token+'.npy')
        final_xy = np.load(raw_dir)   # 8,2
        return Trajectory(final_xy, self._trajectory_sampling)
