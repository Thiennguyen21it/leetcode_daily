from typing import List


class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        first = float("inf")
        second = float("inf")
        for num in nums:
            if num > second:
                return True
            elif num > first and num < second:
                second = num
            elif num < first:
                first = num

        return False


if "__name__" == "__main__":
    solution = Solution()
    nums = [1, 2, 3, 4, 5]
    print(solution.increasingTriplet(nums))
