from typing import List

class Solution:
    def checkEqualPartitions (self, nums: List[int], target: int) -> bool:
        if len(nums) < 2:
            return False 
        
        total_product = 1
        zero_count = 0 

        for num in nums:
            if num == 0:
                zero_count +=1
            else:
                total_product *= num 

        # case 1: target is 0 
        if target == 0:
            if zero_count < 2: # need at least two zero for both subset to have product = 0
                return False 
            
            return total_product == 1
        
        # case 2: for non-zero target 
        if zero_count > 0:
            # if any zero exists, a subset product become 0
            return False 
        
        if total_product != target * target:
            return False
        
        # memorizaton cache: (idx, current_product, is_subset_non_empty)
        memo = {}

        def can_split(idx, curr_product, is_non_empty):
            if idx == len(nums):
                # check if curr subset is non-empty and has product equal to target
                # so, other subset's product is total product / curr_product
                return is_non_empty and curr_product == target and total_product // curr_product == target

            state = (idx, curr_product, is_non_empty)
            
            if state in memo:
                return memo[state]
            
            # include nums[idx] in the first subset 
            include = can_split(idx + 1, curr_product * nums[idx], True)
            
            if include:
                memo[state] = True
                return True 
            
            #exclude nums[idx] from the first subset ( put it in the other subset)
            exclude = can_split(idx + 1, curr_product, is_non_empty)
            
            if exclude:
                memo[state] = True     
                return True 

            memo[state] = False 

            return False

        return can_split(0,1,False)


if __name__ == "__main__" : 

    solution = Solution()
    nums = [3,1,6,8,4]
    target = 24
    
    print("Output:" ,solution.checkEqualPartitions(nums,target))

