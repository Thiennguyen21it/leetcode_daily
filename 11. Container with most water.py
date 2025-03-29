from typing import List

class Solution:
    def maxArea(self, height: List[int]) -> int:
        """
        we have an array height it represent i th of this array is this max heigh of container

        maximm amount of water container is :a * b (a is height ) and b is the width 
        max height => choose minimum distant betwwen to line 
        max width 

        """
        left = 0 
        right = len(height) - 1
        max_volume = 0

        while left < right:
            # height * width
            current_volume = min(height[right], height[left]) * (right - left)
            max_volume = max(max_volume,current_volume)

            if height[left] < height[right]:
                left +=1
            else:
                right -=1
                
        return max_volume 

if __name__ == "__main__":
    solution = Solution()
    # height = [1,1]
    height = [1,8,6,2,5,4,8,3,7]
    print(solution.maxArea(height))

