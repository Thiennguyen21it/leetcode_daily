import numpy as np

def levenshtein_distance(s1, s2):
    """
    Tính khoảng cách Levenshtein (minimum edit distance) giữa hai chuỗi.
    """
    # Chuyển đổi chuỗi thành chữ thường để tìm kiếm không phân biệt hoa thường
    s1 = s1.lower()
    s2 = s2.lower()
    
    # Tạo ma trận khoảng cách
    rows = len(s1) + 1
    cols = len(s2) + 1
    distance = np.zeros((rows, cols), dtype=int)
    
    # Khởi tạo giá trị cho ma trận
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
        
    # Tính toán khoảng cách
    for i in range(1, rows):
        for j in range(1, cols):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1
            distance[i][j] = min(
                distance[i-1][j] + 1,      # xóa
                distance[i][j-1] + 1,      # chèn
                distance[i-1][j-1] + cost  # thay thế
            )
    
    return distance[rows-1][cols-1]

def search_news_titles(keyword, titles, top_n=10):
    """
    Tìm kiếm các tiêu đề tin tức gần đúng nhất với từ khóa.
    
    Args:
        keyword: Từ khóa tìm kiếm
        titles: Danh sách các tiêu đề tin tức
        top_n: Số lượng kết quả trả về
        
    Returns:
        Danh sách top_n tiêu đề có khoảng cách nhỏ nhất với từ khóa
    """
    # Tính khoảng cách cho mỗi tiêu đề
    distances = [(title, levenshtein_distance(keyword, title)) for title in titles]
    
    # Sắp xếp theo khoảng cách tăng dần
    distances.sort(key=lambda x: x[1])
    
    # Trả về top_n kết quả
    return distances[:top_n]

def load_news_titles(file_path):
    """
    Đọc danh sách tiêu đề tin tức từ tập tin.
    
    Args:
        file_path: Đường dẫn đến tập tin chứa tiêu đề tin tức
        
    Returns:
        Danh sách các tiêu đề tin tức
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            titles = [line.strip() for line in file if line.strip()]
        return titles
    except Exception as e:
        print(f"Lỗi khi đọc tập tin: {e}")
        return []

def main():
    # Đường dẫn đến tập tin chứa tiêu đề tin tức
    # file_path = input("Nhập đường dẫn đến tập tin chứa tiêu đề tin tức: ")
    
    # Đọc danh sách tiêu đề
    titles = load_news_titles("./news_titles.txt")
    
   # if not titles:
   #      print("Không thể đọc tiêu đề từ tập tin hoặc tập tin rỗng.")
   #      return
   #
    print(f"Đã đọc {len(titles)} tiêu đề tin tức.")
    #
    # # Nhập từ khóa tìm kiếm
    keyword = input("Nhập từ khóa tìm kiếm: ")
    
    # Tìm kiếm tiêu đề gần đúng
    results = search_news_titles(keyword, titles)
    
    # Hiển thị kết quả
    print("\nKết quả tìm kiếm cho từ khóa:", keyword)
    print("-" * 60)
    for i, (title, distance) in enumerate(results, 1):
        print(f"{i}. {title} (Khoảng cách: {distance})")

if __name__ == "__main__":
    main()
