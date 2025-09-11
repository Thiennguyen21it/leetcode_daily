from typing import List

"""
    Leetcode 394. Decode String
    Given an encode string, return its decoded string.
    The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being 
    reqeated exactly k time. Note that k is guaranteed to be positive integer

    You may assume that the input string is always valid; there no extra white spaces, square brackets are well-formed
    Furthermore, you may assume that the ogirinal data does not contain any digits and that digits are only for those reqeat number,
    k, For example, there will be not be input like 3a or 2[4].
    The test cases are generated so that the length of the output will never exceed 10^5.
 
"""

class Solution:
    def decodeString(self, s:str) -> str:
        """
        Args:
            s: input string need to be encode
        Return:
            result string after being encoded
        """
        stack = []
        
        for i in s:
            if i != ']':
                stack.append(i)
            else:
                # extract substring to be multiple
                curr_str = ""
                while stack[-1] != '[':
                    curr_str = stack.pop() + curr_str
                
                # pop to remove [ bracket
                stack.pop()
                # extract number 
                curr_num = ""
                while stack and stack[-1].isdigit():
                    curr_num = stack.pop() + curr_num
                
                #updating substring to multiple with number
                curr_str = int(curr_num) * curr_str
                
                stack.append(curr_str)

        return "".join(stack)


if __name__ == '__main__':
    solution = Solution()
    s ="100[leetcode]"
    s1 = "3[a]2[bc]" 
    s2 = "3[a2[c]]"
    s3 = "2[abc]3[cd]ef"

    print(solution.decodeString(s)) # expected: "aaabcbc"
    print(solution.decodeString(s2)) # expected: "accaccacc"
    print(solution.decodeString(s3)) # expected: "abcabccdcdcdef"