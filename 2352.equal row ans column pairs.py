from typing import List 
from collections import Counter
class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        count = 0 
        row_count = Counter(tuple(row) for row in grid)
        for col in zip(*grid):
            count += row_count[col]
        return count    

if __name__ == "__main__":
    grid = [[3,2,1],[1,7,6],[2,7,7]]
    print(Solution().equalPairs(grid))