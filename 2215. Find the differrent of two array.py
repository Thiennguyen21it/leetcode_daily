from typing import List 

class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        """
        result of this function is a list of two lists.
        the first list is the difference of nums1 and nums2.
        the second list is the difference of nums2 and nums1.
        """
        result = [[], []]
        result[0] = list(set(nums1) - set(nums2))
        result[1] = list(set(nums2) - set(nums1))
        return result

if __name__ == "__main__":
    solution = Solution()
    nums1 = [1,2,3,3]
    nums2 = [1,1,2,2]
    print(solution.findDifference(nums1, nums2))


