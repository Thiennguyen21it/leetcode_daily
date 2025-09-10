from typing import List

"""
    leetcode 735. Asteroid collision
    We are given an array asteroids of integers representing asteroids in a row. The indices of the asteriod in the array represent their relative position in space.

    For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

    Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.
"""

class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        """
        Args: 
            asteroids: list of integers string.
        Returns:
            Ouput: list of integers string ( rest asteroids after collision) 
        """
        
        """
        We have list of asteroid have the absolute value present its size
        and the size have: postive value ( it mean direction, go left to right -> ) ,otherwise negative is right to left <-
        and we need to compare the abs value that smaller value need destroy.
        Solution:
            1. We inital stack to store asteroid
            2. We loop through each element in list asteroids
            4. We check the current asteroid and the last asteroid in the stack to determine whether there will be a collision.
                that depends on the direction of two asteroid
            5. we define the while loop and check if the direction of current asteroid opposite with last asteroid in list
                asteroid < 0 < stack[-1] (last element in stack)
                first abs value of asteroid > stack[-1]: we destroy the last element in stack and continue
                if abs value of asteroid == stack[-1]: we alse destroy the last element
                otherwise we append the asteroid into list
        """
        stack = []

        for a in asteroids:
            while stack and a < 0 < stack[-1]:
                if abs(a) > stack[-1]:
                    stack.pop()
                    continue
                elif abs(a) == stack[-1]:
                    stack.pop
                break
            else:
                stack.append(a)
        
        return stack

if __name__ == "__main__":
    solution = Solution()
    # test case
    asteroids = [5,10,-5] 
    # expected output : [5,10] : The 10 and -5 collide resulting in 10. The 5 and 10 never collide.
    asteroidCol = solution.asteroidCollision(asteroids)
    print("output:",asteroidCol)
    