class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        # two pointer approach
        i = 0
        j = 0

        while i < len(s) and j < len(t):
            # check char in s vs t match or not
            if s[i] == t[j]:
                i += 1
            j += 1

        return i == len(s)


if __name__ == "__main__":
    solution = Solution()
    s = "abc"
    t = "ahbgdc"
    print("Is Seubsequence :", solution.isSubsequence(s, t))
