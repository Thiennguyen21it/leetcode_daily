from typing import List

class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        left_sum = 0

        for i in range(len(nums)):
            right_sum = total_sum - nums[i] - left_sum

            if right_sum == left_sum:
                return i 
            
            left_sum += nums[i]

        return - 1

if __name__ == '__main__':

    solution = Solution()
    nums = [1,7,3,6,5,6]
    
    print("The pivotIndex is:", solution.pivotIndex(nums))
