from typing import List

class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        n = len(nums)

        zero_count = 0 
        for right in nums:
            
            














if __name__ == "__main__":
    solution = Solution()
    nums = [1,1,0,1]
    longestSubarray = solution.longestSubarray(nums)
    print('Output:', longestSubarray)
