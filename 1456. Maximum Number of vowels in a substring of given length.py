class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        """
        Question: Given a string s and integer k, find the maximum number of vowels in any k-length substring.
        Example: s = "abciiidef", k = 3 -> Answer is 3 (substring "cii" has 3 vowels: i, i, i).
        
        Approach:
        1. Create a set of vowels (a, e, i, o, u) to check characters quickly.
        2. Count vowels in the first k characters (first window).
        3. Slide the window one step at a time:
           - Remove the leftmost character (if it’s a vowel, subtract 1).
           - Add the next character (if it’s a vowel, add 1).
           - Keep track of the maximum vowel count seen.
        """
        # Step 1: Define vowels
        vowel_letters = {'a', 'e', 'i', 'o', 'u'}       
        # vowel_letters = set(aeiou)
        # Step 2: Count vowels in the first window (first k characters)
        vowel_count = 0
        for i in range(k):  # Look at s[0] to s[k-1]
            if s[i] in vowel_letters:
                vowel_count += 1
        max_vowels = vowel_count  # Store the maximum vowel count seen so far
        
        # Step 3: Slide the window across the string
        for i in range(len(s) - k):  # Start at 0, stop when window can't fit
            # Remove the character that’s no longer in the window (s[i])
            if s[i] in vowel_letters:
                vowel_count -= 1
            # Add the new character entering the window (s[i + k])
            if s[i + k] in vowel_letters:
                vowel_count += 1
            # Update the maximum if the current window has more vowels
            if vowel_count > max_vowels:
                max_vowels = vowel_count
        
        # Step 4: Return the maximum number of vowels found
        return max_vowels

if __name__ == "__main__":
    s = "abciiidef"
    k = 3
    solution = Solution()
    print("output:", solution.maxVowels(s,k))
