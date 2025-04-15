from typing import List

class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
      
        """
         
        ques: find contiguous subarray whose length is equal to given number k and return maximum average value of this subarray 
           
        the first solution i have is in nums array : i iterate from 0 to k 
        and caculate average of each value in this

        nums = [1,12,-5,-6,50,3]


        i = 0 to k: 1,12,-5,-6 => caculate current average from 0 to k  

        i = 1 to k,  12,-5,-6,50 => caculate average and check if max average 

        i= 2 to k , -5,-6,50,3 => ...

        loop until not enough k 

        we minus first element index of window and plus next k +1 index
        
        """
       
        #first we caculate sum from 0 to k in array nums 
        
        window_sum = sum(nums[:k])
        max_sum = window_sum

        #we iterate until out of range k index in the array 
        for i in range(0, len(nums) - k):
            #we caculate sum of window with minus first element and plus k +1
            window_sum = window_sum - nums[i] + nums[k] 

            max_sum = max(max_sum, window_sum)

        #we return max average base on max_sum / k 
        return max_sum / k


if __name__ == "__main__":
    solution = Solution()

    nums = [1,12,-5,-6,50,3]
    k = 4
    
    print("maximum average:",solution.findMaxAverage(nums,k))
