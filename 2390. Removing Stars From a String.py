from typing import List
"""
    Leetcode 2390. Removing Stars From a String (medium)
    You are given a string s, which contains stars *.

    In one operation, you can:

    Choose a star in s.
    Remove the closest non-star character to its left, as well as remove the star itself.
    Return the string after all stars have been removed.
"""
class Solution:
    # solution 1
    def removeStars(self, s: str) -> str:
        """
        Args: 
        s (str): input string containing '*' characters.

        Returns:
        str: the final string after all '*' removals.

        step-by-step solution using stack.
            1.initial an empty list (stack) result that will be used to build the modified string
            2.loop through each char in the input string s.
            3.if current char i is asterisk (*) , remove the last element from the result list
            5.join the element in the result list into string and return result.
        
        Time Complexity: O(n) : because each character is processed once (added to or removed from the stack)
        Space Complexity: O(n): the worst case (string without '*'), we need to store the entire char in the stack.
        """
        result = []
        asterisk = '*'
        for char in s:
            if char == asterisk:
                result.pop()
            else:
                result.append(char)
        
        return "".join(result)
    
    #soluton 2:
    """
        1. Initial skip_variable that count the number of characters to skip (corresspoding to the number of '*' encountered)
        2. Loop through the string left to right:
            if encounter '*' , increased skip by 1.
            otherwise, if skip > 0, we skip this character and decreased skip by 1
                       if skip == 0, we keep this char
        3. reversed the result string

        Time Complexity: O(n)
        Space Complexity: O(1)
    """
    def removeStars2(self, s: str) -> str:
        """
        Args: 
        s (str): input string containing '*' characters.

        Returns:
        str: the final string after all '*' removals.
        """
        result = []
        skip = 0
        
        # loop right to left
        for char in reversed(s):
            if char == '*':
                skip +=1
            elif skip > 0:
                skip -=1
            else:
                result.append(char)

        return "".join(reversed(result))


if __name__ == '__main__':
    # Input
    s = "leet**cod*e"
    # Expected Output: "lecoe"
    solution = Solution()
    print(solution.removeStars2(s))