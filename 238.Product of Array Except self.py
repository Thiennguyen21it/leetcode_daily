from typing import List
class Solution:
    def productArray(self, nums:List[int]) -> List[int]:

nums = [1, 2, 3, 4]
n = len(nums)
output = [1] * n
for i in range(1, n):
    output[i] = nums[i - 1] * output[i - 1]

right = 1

for i in range(n - 1, -1, -1):
    output[i] = nums[i] * right
    right *= nums[i]
print(output)
