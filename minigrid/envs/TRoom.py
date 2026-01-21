from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Floor
from minigrid.core.mission import MissionSpace
from gymnasium import spaces
import numpy as np


class TRoom_Env(MiniGridEnv):
    """
    T-room environment in the *minigrid* (modern) style, matching File 1's structure:
    - MissionSpace + _gen_mission
    - kwargs plumbing for max_steps
    - action_space = Discrete(4)
    - see_through_walls=True
    - agent_start_pos/dir handling identical to File 1
    - same object placement logic as File 2 (order + colors + locations)
    - same T-room wall logic as File 2
    """

    def __init__(
        self,
        size=16,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        tri_color="blue",
        plus_color="red",
        x_color="yellow",
        order="TPXD",
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.tri_color = tri_color
        self.plus_color = plus_color
        self.x_color = x_color
        self.order = order
        self.size = size

        mission_space = MissionSpace(mission_func=self._gen_mission)

        # Match File 1 behavior: pop max_steps if provided, otherwise default
        max_steps = kwargs.pop("max_steps", 10 * size * size)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            **kwargs,
        )

        # Match File 1: only 4 actions
        self.action_space = spaces.Discrete(4)

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height, regenerate):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Surrounding walls (same as File 2 / File 1 style)
        self.grid.horz_wall(0, 0)
        self.grid.vert_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(width - 1, 0)

        # --- Shapes: same locations + order logic as File 2 ---
        loc = [
            (width / 3 - 4, height / 3 - 4),
            (2 * width / 3 - 2, height / 3 - 4),
            (width / 3 - 3, 2 * height / 3 - 2),
            (2 * width / 3 - 2, 2 * height / 3 - 2),
        ]

        shapes = {
            "T": {"name": "triangle", "color": self.tri_color},
            "P": {"name": "plus", "color": self.plus_color},
            "X": {"name": "x", "color": self.x_color},
            "D": {"name": "dash", "color": self.tri_color},
        }

        for idx, char in enumerate(self.order):
            self.place_shape(shapes[char]["name"], loc[idx], shapes[char]["color"])

        self.mission = "get to the green goal square"

        # --- T-room walls: same logic as File 2 ---
        TRoom_delimeter = 5
        if self.size == 18:
            TRoom_delimeter = 6
        elif self.size == 20:
            TRoom_delimeter = 7

        for i in range(TRoom_delimeter):
            self.grid.vert_wall(i, int(width / 2), length=int(width / 2))
            self.grid.vert_wall(int(height - 2) - i, int(width / 2), length=int(width / 2))

        for j in range(int(TRoom_delimeter / 2) + 2):
            self.grid.horz_wall(0, j, length=width - 1)

        # Place the agent (match File 1)
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def place_shape(self, shape, pos, color):
        """
        Place a 6x6 shape with lower left corner at (x,y)
        """
        shapegrid = {
            "plus": np.array(
                [
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                ]
            ),
            "triangle": np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                ]
            ),
            "x": np.array(
                [
                    [1, 1, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 1, 1],
                ]
            ),
            "dash": np.array(
                [
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                ]
            ),
        }

        shapecoords = np.transpose(np.nonzero(shapegrid[shape])) + np.array(pos, dtype="int32")
        for coord in shapecoords:
            self.put_obj(Floor(color), int(coord[0]), int(coord[1]))


class TRoomEnv_16(TRoom_Env):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None, order="TPXD", **kwargs)


class TRoomEnv_18(TRoom_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18, agent_start_pos=None, order="TPXD", **kwargs)


class TRoomEnv_20(TRoom_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20, agent_start_pos=None, order="TPXD", **kwargs)
