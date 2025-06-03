from typing import List

class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        zero_count = 0 
        left = 0 
        longestWindow = 0 

        for right in range(len(nums)):
            if nums[right] == 0:
                zero_count +=1

            while zero_count > 1:
                if nums[left] == 0:
                    zero_count -=1
                left +=1

            longestWindow = max(longestWindow, right - left)

        return longestWindow

if __name__ == "__main__":
    solution = Solution()
    nums = [1,1,0,1]
    longestSubarray = solution.longestSubarray(nums)
    print('Output:', longestSubarray)
