from typing import List

class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        hightest_altitude = 0   
        current_altitude = 0

        for i in range(0, len(gain)):
            current_altitude += gain[i]
            hightest_altitude = max(hightest_altitude, current_altitude)

        return hightest_altitude

if __name__ == "__main__":
    solution = Solution()
    gain = [-5,1,5,0,-7]
    
    print("The hightest altitude is:", solution.largestAltitude(gain))
