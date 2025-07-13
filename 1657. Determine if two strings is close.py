from collections import Counter
from typing import List 

class Solution: 
    def closeStrings(self, word1: str, word2: str) -> bool: 
        if len(word1) != len(word2): 
            return False 
        
        count1 = Counter(word1)
        count2 = Counter(word2)

        if count1.keys() != count2.keys(): 
            return False 
        
        if sorted(count1.values()) != sorted(count2.values()):
            return False
        
        return True 

if __name__ == "__main__": 
    solution = Solution()
    print(solution.closeStrings("abc", "bca"))
    print(solution.closeStrings("a", "aa"))
    print(solution.closeStrings("cabbba", "abbccc"))
    print(solution.closeStrings("cabbba", "aabbss"))
    print(solution.closeStrings("uau", "ssx"))
    