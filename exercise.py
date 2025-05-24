import numpy as np
from typing import List
from distancia import levenshtein_distance
def levenshtein_distance(s1: str, s2: str) -> int:
    """Tính khoảng cách Levenshtein giữa hai chuỗi."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def load_news_titles(filename: str) -> List[str]:
    """Đọc tiêu đề tin tức từ file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Không tìm thấy file {filename}")
        return []

def find_similar_titles(keyword: str, titles: List[str], num_results: int = 10) -> List[tuple]:
    """Tìm các tiêu đề gần giống nhất với từ khóa."""
    distances = []
    for title in titles:
        dist = levenshtein_distance(keyword.lower(), title.lower())
        distances.append((dist, title))
    
    # Sắp xếp theo khoảng cách tăng dần
    distances.sort(key=lambda x: x[0])
    return distances[:num_results]

def main():
    # Đường dẫn đến file chứa tiêu đề tin tức
    filename = "news_titles.txt"
    
    # Đọc danh sách tiêu đề
    titles = load_news_titles(filename)
    if not titles:
        return
    
    while True:
        keyword = input("\nNhập từ khóa tìm kiếm (hoặc 'q' để thoát): ")
        if keyword.lower() == 'q':
            break
            
        similar_titles = find_similar_titles(keyword, titles)
        print(f"\nTop 10 tiêu đề gần giống nhất với '{keyword}':")
        for dist, title in similar_titles:
            print(f"Khoảng cách: {dist} - {title}")

if __name__ == "__main__":
    main()