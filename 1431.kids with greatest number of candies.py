from typing import List


class Solution:
    def kidsWithCandies(self, candies: List[int], extraCanies: int) -> List[bool]:
        # result = []
        # for i in range(0, len(candies)):
        #     if candies[i] + extraCanies >= max(candies):
        #         result.append(True)
        #     else:
        #         result.append(False)
        # return result
        #
        return [candy + extraCanies >= max(candies) for candy in candies]


if __name__ == "__main__":
    solution = Solution()
    candies = [2, 3, 5, 1, 3]
    extraCanies = 3
    print(solution.kidsWithCandies(candies, extraCanies))
    print("heelo")
