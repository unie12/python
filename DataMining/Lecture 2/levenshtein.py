def levenshtein_distance(t1, t2):
    rows, colums = len(t1) + 1, len(t2) + 1
    arr = [
        list(range(colums)) if not i else [i] + [0] * (colums-1)
        for i in range(rows)
    ]

    for i in range(1, rows):
        for j in range(1, colums):
            if t1[i-1] == t2[j-1]:
                arr[i][j] = arr[i-1][j-1]
            else:
                arr[i][j] = min(
                    arr[i-1][j],
                    arr[i][j-1],
                    arr[i-1][j-1],
                ) + 1
    for r in arr:
        print(r)
    return arr[rows-1][colums-1]

print(levenshtein_distance('abcd', 'abecd'))
print(levenshtein_distance('가나다라', '가마바라사'))
print(levenshtein_distance('영광', '광영'))