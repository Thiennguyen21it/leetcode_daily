from typing import List


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        # idea: swap nums in array to left side and it mean 0 go to right side
        left = 0
        for i in range(0, len(nums)):
            if nums[i] != 0:
                nums[left], nums[i] = nums[i], nums[left]
                left += 1


if __name__ == "__main__":
    nums = [0, 1, 0, 3, 12]
    solution = Solution()
    print("Move Zeros:", solution.moveZeroes(nums))
