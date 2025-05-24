class Solution:
    def reverseWords(self, s: str) -> str:
        # Split the string into words, removing extra spaces
        words = s.split()
        # Reverse the list of words
        reversed_words = words[::-1]
        # Join the reversed words with a single space
        return ' '.join(reversed_words)
"""

### Giải thích:
1. **`s.split()`**: Tách chuỗi `s` thành danh sách các từ, tự động loại bỏ khoảng trắng thừa.
2. **`words[::-1]`**: Đảo ngược danh sách các từ.
3. **`' '.join(reversed_words)`**: Ghép các từ đã đảo ngược thành chuỗi, sử dụng một khoảng trắng giữa các từ.
"""

if __name__ == "__main__":
    solution = Solution()
    print(solution.reverseWords("  hello world  "))  # Output: "world hello"
    print(solution.reverseWords("a good   example"))  # Output: "example good a"
