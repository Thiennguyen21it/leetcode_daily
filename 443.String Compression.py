from typing import List


class Solution:
    def compress(self, chars: List[str]) -> int:
        write = 0
        read = 0

        while read < len(chars):
            char = chars[read]
            count = 0

            # count the number of ocurrences of curr char
            while read < len(chars) and chars[read] == char:
                read += 1
                count += 1

            # write the char to the write position
            chars[write] = char
            write += 1

            # if the count is > 1, write the count as well
            if count > 1:
                for i in str(count):
                    chars[write] = i
                    write += 1

        return write


if __name__ == "__main__":
    solution = Solution()
    chars = ["a", "a", "b", "b", "c", "c", "c"]
    print("length of new array: ", solution.compress(chars))
    print("return new array chars: ", chars)
