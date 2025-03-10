from typing import List


class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        # solution 1: two pointers
        i = 0
        len_flowerbed = len(flowerbed)

        while i < len_flowerbed:
            if flowerbed[i] == 0:
                left_empty = i == 0 or (flowerbed[i - 1] == 0)
                right_empty = len_flowerbed - 1 == 0 or (flowerbed[i + 1] == 0)

                if left_empty and right_empty:
                    n -= 1
                    flowerbed[i] = 1
                    i += 1

            i += 1

        return n <= 0

    def canPlaceFlowers2(self, flowerbed: List[int], n: int) -> bool:
        # solution2:
        new_flowerbed = [0] + flowerbed + [0]

        for i in range(1, len(new_flowerbed) - 1):
            if new_flowerbed[i] == 1:
                continue
            if (new_flowerbed[i == 1] == 0 and new_flowerbed[i + 1] == 0) and n > 0:
                new_flowerbed[i] == 1
                n -= 1
        return n == 0


if __name__ == "__main__":
    solution = Solution()
    # print(solution.canPlaceFlowers([1, 0, 0, 0, 1], 1))
    print(solution.canPlaceFlowers2([1, 0, 0, 0, 1], 1))
