from typing import List

class Solution:
    def longestOnes (self, nums: List[int], k: int) -> int :
        """
        this func is finding longest subarray contain all 1 number 
        with the condition that it is possible to flip at most k 0s to 1s
        
        """
        
        left = 0
        zero_count = 0 
        max_len = 0

        for right in range(len(nums)):
            if nums[right] == 0:
                zero_count += 1

            while zero_count > k:
                if nums[left] == 0:
                    zero_count -=1
            left +=1

            curr_len = right - left + 1 

            max_len = max(max_len, curr_len)

        return max_len

if __name__ == "__main__":
    solution = Solution()

    nums = [1,1,1,0,0,0,1,1,1,1,0]
    k = 2
    print(nums)
    print(solution.longestOnes(nums,k))
