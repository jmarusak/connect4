class Board:
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}
        
    def drop_stone(self, player, point):
        self._grid[point] = player

    def get(self, point):
        return self._grid.get(point)
