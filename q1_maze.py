# q1_maze.py
from typing import List, Tuple
import heapq


def astar_maze(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    grid: List[List[str]]
) -> List[Tuple[int, int]]:
    """
    题目一：迷宫寻路。
    返回从 start 到 goal 的最短路径坐标列表；无解返回 []。
    """
    if not grid or not grid[0]:
        return []

    rows, cols = len(grid), len(grid[0])

    def passable(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] != 'X'

    if not passable(*start) or not passable(*goal):
        return []

    def heuristic(pos: Tuple[int, int]) -> int:
        # 曼哈顿距离
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_heap = [(heuristic(start), 0, start)]
    g_cost = {start: 0}
    parent = {start: None}
    closed = set()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_heap:
        _, cur_g, cur = heapq.heappop(open_heap)

        if cur in closed:
            continue

        if cur == goal:
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]

        closed.add(cur)
        r, c = cur

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)

            if not passable(nr, nc):
                continue

            new_g = cur_g + 1
            if new_g < g_cost.get(nxt, 10**18):
                g_cost[nxt] = new_g
                parent[nxt] = cur
                f = new_g + heuristic(nxt)
                heapq.heappush(open_heap, (f, new_g, nxt))

    return []


if __name__ == "__main__":
    start = (0, 0)
    goal = (3, 4)
    grid = [
        ['S', ' ', ' ', 'X', ' '],
        [' ', 'X', ' ', ' ', ' '],
        [' ', 'X', ' ', ' ', ' '],
        [' ', 'X', ' ', ' ', ' '],
        [' ', 'X', ' ', ' ', ' '],
    ]

    path = astar_maze(start, goal, grid)
    print("题目一输出：")
    print(path)