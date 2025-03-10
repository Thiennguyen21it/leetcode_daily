word1 = "abc"
word2 = "pgr"

result = []
for i in range(max(len(word1), len(word2))):
    if i < len(word1):
        result.append(word1[i])
    if i < len(word2):
        result.append(word2[i])

result = "".join(result)
print(result)
