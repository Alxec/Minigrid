from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Floor
from minigrid.core.mission import MissionSpace
from gymnasium import spaces
import numpy as np


class Donut_Env(MiniGridEnv):
    """
    Donut-room environment in modern *minigrid* style (matching File 1 structure),
    but with the *exact* _gen_grid (walls + objects) from the DonutLapRoom snippet you sent.
    """

    def __init__(
        self,
        size=16,
        Lwidth=10,
        Lheight=8,
        agent_start_pos=(3, 3),
        agent_start_dir=0,
        tri_color="blue",
        plus_color="red",
        x_color="yellow",
        order="TPXD",
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.Lwidth = Lwidth
        self.Lheight = Lheight
        self.tri_color = tri_color
        self.plus_color = plus_color
        self.x_color = x_color
        self.order = order

        # Keep this like the snippet (and File 1)
        see_through_walls = True

        # Mirror the snippet's naming
        self.start_pos = agent_start_pos
        self.size = size

        mission_space = MissionSpace(mission_func=self._gen_mission)
        max_steps = kwargs.pop("max_steps", 10 * size * size)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            **kwargs,
        )

        # Match File 1
        self.action_space = spaces.Discrete(4)

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height, regenerate=True):
        # Exact regenerate behavior from your snippet
        if not regenerate:
            if self.start_pos is not None:
                self.agent_pos = self.start_pos
                self.agent_dir = self.agent_start_dir
            else:
                self.place_agent()
            return

        self.grid = Grid(width, height)

        # Outer walls (exact)
        self.grid.horz_wall(0, 0)
        self.grid.vert_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(width - 1, 0)

        # Donut wall (horizontal bar) (exact)
        for i in range(int(height / 2) - 4, int(height / 2) + 4):
            self.grid.horz_wall(int(self.Lwidth / 2), i, length=8)

        # Place agent (exact)
        if self.start_pos is not None:
            self.agent_pos = self.start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place the four ordered shapes (exact)
        loc = [
            (width / 3 - 4, height / 3 - 4),
            (2 * width / 3 - 1, height / 3 - 1),
            (width / 3 - 3, 2 * height / 3 - 2),
            (2 * width / 3 - 2, 2 * height / 3 - 2),
        ]

        shapes = {
            "T": {"name": "triangle", "color": self.tri_color},
            "P": {"name": "dash", "color": self.plus_color},
            "X": {"name": "x", "color": self.x_color},
            "D": {"name": "dash", "color": self.tri_color},
        }

        for idx, char in enumerate(self.order):
            self.place_shape(shapes[char]["name"], loc[idx], shapes[char]["color"])

        # Additional decorations (exact)
        self.place_shape("plus", (width / 3 - 1, height / 3 - 5), self.x_color)
        self.place_shape("plus", (width / 3,     height / 3 - 5), self.x_color)
        self.place_shape("plus", (width / 3 + 1, height / 3 - 5), self.x_color)
        self.place_shape("plus", (width / 3 + 2, height / 3 - 5), self.x_color)

        self.place_shape("plus", (width / 3 - 3, height / 3 + 6), self.plus_color)
        self.place_shape("plus", (width / 3 - 2, height / 3 + 6), self.plus_color)
        self.place_shape("plus", (width / 3 - 1, height / 3 + 6), self.plus_color)
        self.place_shape("plus", (width / 3,     height / 3 + 6), self.plus_color)

        self.mission = self._gen_mission()

    def place_shape(self, shape, pos, color):
        """
        Exact shape definitions from your snippet.
        Place a 6x6 shape with lower left corner at (x,y)
        """
        shapegrid = {
            "plus": np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                ]
            ),
            "triangle": np.array(
                [
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            "x": np.array(
                [
                    [0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ),
            "dash": np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                ]
            ),
        }

        shapecoords = np.transpose(np.nonzero(shapegrid[shape])) + np.array(pos, dtype="int32")
        for coord in shapecoords:
            self.put_obj(Floor(color), int(coord[0]), int(coord[1]))


class DonutEnv_16(Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None, **kwargs)


class DonutEnv_18(Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18, agent_start_pos=None, **kwargs)


class DonutEnv_20(Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20, agent_start_pos=None, **kwargs)


import numpy as np
from minigrid.core.world_object import Wall, Floor, COLOR_TO_IDX, COLORS
from minigrid.core.grid import Grid

# Registering your specific "Maximally Different" palette
custom_colors = {
    "cyan":       np.array([0, 255, 255]),
    "magenta":    np.array([255, 0, 255]),
    "white":      np.array([255, 255, 255]),
    "lime":       np.array([50, 205, 50]),
    "orange":     np.array([255, 165, 0]),
    "black":      np.array([0, 0, 0])
}

for name, rgb in custom_colors.items():
    if name not in COLOR_TO_IDX:
        COLOR_TO_IDX[name] = len(COLOR_TO_IDX)
    COLORS[name] = rgb

class CustomColorDonutEnv(Donut_Env):
    def __init__(self, **kwargs):
        # Mapping the new colors to the shape attributes
        super().__init__(
            tri_color="cyan",    # Triangle
            plus_color="white",   # Dash/Plus
            x_color="lime",      # X
            **kwargs
        )

    def _gen_grid(self, width, height, regenerate=True):
        self.grid = Grid(width, height)

        # 1. WALLS (Black)
        w_col = "black"
        for i in range(width):
            self.grid.set(i, 0, Wall(w_col))
            self.grid.set(i, height - 1, Wall(w_col))
        for i in range(height):
            self.grid.set(0, i, Wall(w_col))
            self.grid.set(width - 1, i, Wall(w_col))

        # Donut wall
        for i in range(int(height / 2) - 4, int(height / 2) + 4):
            for j in range(8):
                self.grid.set(int(self.Lwidth / 2) + j, i, Wall(w_col))

        # 2. AGENT (Place before floor to avoid the infinite loop)
        if self.start_pos is not None:
            self.agent_pos = self.start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # 3. OBJECTS (Cyan, White, Lime)
        loc = [
            (width / 3 - 4, height / 3 - 4),
            (2 * width / 3 - 1, height / 3 - 1),
            (width / 3 - 3, 2 * height / 3 - 2),
            (2 * width / 3 - 2, 2 * height / 3 - 2),
        ]
        shapes = {
            "T": {"name": "triangle", "color": self.tri_color},
            "P": {"name": "dash", "color": self.plus_color},
            "X": {"name": "x", "color": self.x_color},
            "D": {"name": "dash", "color": self.tri_color},
        }
        for idx, char in enumerate(self.order):
            self.place_shape(shapes[char]["name"], loc[idx], shapes[char]["color"])

        # Extra decorations
        for i in range(4):
            self.place_shape("plus", (width / 3 - 1 + i, height / 3 - 5), self.x_color)
            self.place_shape("plus", (width / 3 - 3 + i, height / 3 + 6), self.plus_color)

        # 4. FLOOR (Magenta)
        # We fill the empty background with Magenta
        for x in range(width):
            for y in range(height):
                if self.grid.get(x, y) is None:
                    self.grid.set(x, y, Floor("magenta"))

        self.mission = self._gen_mission()

# Registration for your 16x16 version
class OrthogonalDonutEnv_16(CustomColorDonutEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None, **kwargs)