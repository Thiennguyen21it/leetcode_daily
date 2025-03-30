from typing import List 
from collections import defaultdict
class Solution:
    def maxOperations(self, nums:List[int], k:int) -> int:
        """
        first approach: use two pointer (o(nlogn)
        we need to sort this array and create two pointer: 1 is begin of array and 1 is end of this
        we check if nums[left] + nums[right] equal to target k , and if this greater than k we decrease right ponter 
        """
        nums.sort()
        left , right = 0 , len(nums) - 1 
        count_pair = 0 

        while left < right:
            if (nums[left] + nums[right]) == k: 
                count_pair +=1
                left +=1
                right -=1 
            elif (nums[left] + nums[right]) < k:
                left +=1
            else:
                right -=1

        return count_pair
    
    def maxOperations2(self, nums: List[int], k:int) -> int:
        """
        second approach is use hasmap to remember number have meet and this number of occurrences ,
        while we check a num in array , we need to find the other number k - num to create a satisfied pair ,
        it means: we store a number and count this like map = { number_3 :  1 } and we have k(6) - 3 = 3 
        we check map[k-3] > 0 ? if true: we increase count_pair and decrease count like map = {number_3: 0 } (1 -> 0)  
        """
        map = defaultdict(int)
        count_pair = 0 
        
        for num in nums:
            if (map[k - num])> 0:
                count_pair +=1
                map[k - num] -=1
            else:
                map[num] +=1
        return count_pair


if __name__ == "__main__":
    solution = Solution()
    nums = [1,2,3,4]
    k = 5 
    print(solution.maxOperations2(nums,k))
