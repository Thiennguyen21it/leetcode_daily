from typing import List


#     max_sum = window
#
#     for i in range(k, len(arr)) :
#         window += arr[i] - arr[i-k]
#         max_sum = max(max_sum,window)
#
#     return max_sum
#
# # arr = [1, 4, 2, 10, 23, 3, 1, 0, 20]
# # k = 4
# # print("max sum:",max_sum_subarray(arr,k))
#
# def min_sum_subarray(arr,k):
#     window = sum(arr[:k])
#     min_sum = window
#
#     for i in range(k, len(arr)):
#         window += arr[i] - arr[i-k]
#         min_sum = min(min_sum,window)
#
#     return min_sum
#
#
# arr = [3, 7, 1, 2, 8, 4, 5]
# k = 3
# print("min sum:",  min_sum_subarray(arr,k))

#Đề bài:

# Cho chuỗi nhị phân (gồm 0 và 1), tìm độ dài lớn nhất của chuỗi con chỉ toàn là 1 có độ dài tối đa là k số 0 được đổi thành 1.
#
# Ví dụ:
#
# s = "110100110"
# k = 2
# Đổi tối đa 2 số 0 thành 1 => kết quả: 5 ("11011" hoặc "01101")
from typing import List
# s = "110100110"
# k = 2
#
# def longestOnes(s, k):
#     max_len = 0 
#     zero_count = 0
#     left = 0 
#     for right in range(len(s)):
#         if s[right] == '0':
#             zero_count +=1
#
#         while zero_count > k:
#             if s[left] == '0':
#                 zero_count -= 1
#             left = left + 1
#
#         max_len = max(max_len, right - left + 1 )
#
#     return max_len
#
# print("longest one:", longestOnes(s,k))
#
        
#
# def longest_zeros(s: str, k: int):
#     zero_max_len = 0 
#     one_count = 0
#     left = 0 
#
#     for right in range(len(s)):
#         if s[right] == "1":
#             one_count +=1
#
#         while one_count > k:
#             if s[left] == "1":
#                 one_count -=1
#
#             left +=1
#
#         zero_max_len = max(zero_max_len, right - left + 1)
#
#     return zero_max_len
#
#
# s = "11000111001"
# k = 2
#
# print("output:", longest_zeros(s,k))
#

# def min_subarray_len(target: int, nums: List[int]):
#     left = 0 
#     total = 0 
#     min_len = float('inf')
#     for right in range(len(nums)):
#         total += nums[right] 
#         # print("total", total)
#         while total >= target:
#             min_len = min(min_len, right - left + 1)
#             total -= nums[left]
#             left +=1 
            
#     return min_len if min_len != float('inf') else 0

# target = 15
# nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(min_subarray_len(target, nums)) 

# print("Hello world")


# stack = []

# stack.append('a')
# stack.append('b')
# stack.append('c')

# print(stack)

# print(stack.pop())

# def is_valid_parentheses(s: str) -> bool:
#     stack = []
#     # mapping = {'(':')', '{':'}','[':']'}
#     mapping = {')': '(', '}': '{', ']': '['}
#     for c in s:
#         if c in mapping.values():
#             stack.append(c)
#         elif c in mapping:
#             if not stack or stack[-1] != mapping[c]:
#                 return False
#             stack.pop 

#     return not stack
# print(is_valid_parentheses("()[]{}"))
mapping = {')': '(', '}': '{', ']': '['}
stack = []

stack.append('(')

if not stack or stack[-1] != mapping[')']:
    print(False) 
stack.pop()

print(stack)


































