from typing import List

class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        count = {}
        for num in arr:
            count[num] = count.get(num,0) + 1
        return len(count.values()) == len(set(count.values()))

if __name__ == "__main__":
    solution = Solution()
    arr = [1,2,2,1,1,3]
    print(solution.uniqueOccurrences(arr))