class Solution:
    def gddOfStrings(self, str1: str, str2: str) -> str:
        def gcd_recursive(a, b):
            if b == 0:
                return a
            if a % b == 0:
                return b
            return gcd_recursive(a, a % b)

        if str1 + str2 != str2 + str1:
            return ""
        gcd_len = gcd_recursive(len(str1), len(str2))
        return str1[:gcd_len]


if __name__ == "__main__":
    str1 = "ABCABC"
    str2 = "ABC"
    solution = Solution()
    print(solution.gddOfStrings(str1, str2))
