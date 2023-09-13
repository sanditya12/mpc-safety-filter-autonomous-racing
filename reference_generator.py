import numpy as np
from typing import List, Tuple


class ReferenceGenerator:
    def __init__(
        self, horizon: int, center_points: List[Tuple[float, float]]
    ) -> None:
        self.horizon = horizon
        self.center_points = center_points

    def generate_map(
        self, position: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        index = self.get_closest_position_index(position, self.center_points)

        return self.wrap_slice(self.center_points, index, self.horizon)

    def wrap_slice(self, t: Tuple, start: int, length: int) -> Tuple:
        # Get the initial slice
        result = t[start : start + length]

        # Check if we've reached or exceeded the end of the tuple
        if len(result) < length:
            required = length - len(result)
            result += t[:required]
            # start = required

        return result

    def get_closest_position_index(
        self, pos: Tuple[float, float], positions: List[Tuple[float, float]]
    ) -> int:
        # Calculate the squared distance to avoid sqrt calculation for efficiency
        def squared_distance(
            p1: Tuple[float, float], p2: Tuple[float, float]
        ) -> float:
            return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

        # Find the index of the position with the minimum squared distance
        return min(
            enumerate(positions), key=lambda x: squared_distance(pos, x[1])
        )[0]
