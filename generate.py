import random
from datetime import datetime, timedelta

def generate_news_titles(num_titles=1000, output_file="news_titles.txt"):
    """
    Tạo và lưu một tập hợp các tiêu đề tin tức ngẫu nhiên vào một tập tin.
    
    Args:
        num_titles: Số lượng tiêu đề cần tạo
        output_file: Tên tập tin đầu ra
    """
    
    # Danh sách các từ khóa chủ đề
    topics = [
        "Chính trị", "Kinh tế", "Xã hội", "Thể thao", "Giáo dục",
        "Y tế", "Công nghệ", "Du lịch", "Môi trường", "Văn hóa",
        "Giải trí", "Pháp luật", "Đời sống", "Khoa học", "Quốc tế",
        "Tài chính", "Bất động sản", "Giao thông", "An ninh", "Việc làm"
    ]
    
    # Danh sách các chủ thể
    subjects = [
        "Việt Nam", "Chính phủ", "Quốc hội", "Bộ Giáo dục", "Bộ Y tế",
        "Thủ tướng", "Chủ tịch nước", "Bộ Công an", "Bộ Tài chính", "Ngân hàng Nhà nước",
        "VinGroup", "FPT", "Viettel", "EVN", "PVN",
        "Hà Nội", "TP.HCM", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
        "Đồng bằng sông Cửu Long", "Miền Trung", "Tây Nguyên", "Đông Nam Bộ", "Tây Bắc",
        "Đội tuyển Việt Nam", "U23 Việt Nam", "CLB Hà Nội FC", "Hoàng Xuân Vinh", "Nguyễn Quang Hải"
    ]
    
    # Danh sách các hành động
    actions = [
        "công bố", "ban hành", "thông qua", "triển khai", "phát động",
        "ra mắt", "ký kết", "tổ chức", "khánh thành", "khai trương",
        "phê duyệt", "đầu tư", "phát triển", "xây dựng", "mở rộng",
        "tăng cường", "thúc đẩy", "nâng cao", "cải thiện", "đổi mới",
        "giải quyết", "khắc phục", "ứng phó", "phòng chống", "ngăn chặn",
        "giành chiến thắng", "đạt thành tích", "lập kỷ lục", "giành huy chương", "đoạt giải"
    ]
    
    # Danh sách các đối tượng
    objects = [
        "dự án", "kế hoạch", "chương trình", "chính sách", "đề án",
        "nghị định", "thông tư", "quyết định", "nghị quyết", "văn bản",
        "hội nghị", "hội thảo", "cuộc họp", "buổi làm việc", "lễ ký kết",
        "khu đô thị", "nhà máy", "tuyến đường", "cầu", "bệnh viện",
        "trường học", "công viên", "khu du lịch", "sân bay", "cảng biển",
        "giải đấu", "cuộc thi", "kỳ thi", "tranh giải", "vòng loại"
    ]
    
    # Danh sách các tính từ mô tả
    adjectives = [
        "quan trọng", "lớn", "mới", "hiện đại", "tiên tiến",
        "đột phá", "chiến lược", "ấn tượng", "nổi bật", "xuất sắc",
        "hiệu quả", "bền vững", "sáng tạo", "đa dạng", "toàn diện",
        "chất lượng", "an toàn", "thuận lợi", "phù hợp", "hấp dẫn"
    ]
    
    # Danh sách các địa điểm
    locations = [
        "tại Việt Nam", "tại Hà Nội", "tại TP.HCM", "tại Đà Nẵng", "tại Cần Thơ",
        "tại các tỉnh miền Trung", "tại khu vực đồng bằng sông Cửu Long", "tại vùng núi phía Bắc", 
        "tại Tây Nguyên", "tại đô thị lớn",
        "tại châu Á", "tại khu vực ASEAN", "tại quốc tế", "tại thị trường nước ngoài", "tại các nước phát triển"
    ]
    
    # Danh sách cụm từ thời gian
    time_phrases = [
        "trong năm 2023", "trong quý đầu năm", "trong 6 tháng đầu năm", "trong thời gian tới", "trong giai đoạn 2021-2025",
        "năm 2023", "năm học 2023-2024", "quý 2/2023", "tháng 6/2023", "đầu năm 2024"
    ]
    
    # Danh sách các cụm từ liên kết
    connecting_phrases = [
        "nhằm mục đích", "hướng tới", "để đạt được", "với mục tiêu", "với kỳ vọng",
        "đánh dấu bước tiến", "tạo đà phát triển", "góp phần vào", "nâng tầm vị thế", "khẳng định vai trò"
    ]
    
    # Danh sách các cụm từ kết quả/ảnh hưởng
    result_phrases = [
        "mang lại lợi ích cho người dân", "thúc đẩy tăng trưởng kinh tế", "cải thiện đời sống người dân",
        "nâng cao chất lượng cuộc sống", "bảo vệ môi trường", "phát triển bền vững",
        "tạo việc làm cho người lao động", "thu hút đầu tư nước ngoài", "tăng cường hợp tác quốc tế",
        "giải quyết các vấn đề xã hội", "ứng phó với biến đổi khí hậu", "đảm bảo an ninh quốc gia"
    ]
    
    # Các mẫu tiêu đề
    title_templates = [
        "{subject} {action} {object} {adjective} {location}",
        "{subject} {action} {object} {time}",
        "{topic}: {subject} {action} {object} {connecting} {result}",
        "{subject} {action} {object} {connecting} {result} {time}",
        "{topic}: {subject} {action} {object} {location} {time}",
        "{subject} {action} {adjective} {object} {location}",
        "{topic} {location} có nhiều chuyển biến {adjective} {time}",
        "{subject} dự kiến {action} {object} {adjective} {time}",
        "{object} {adjective} được {action} bởi {subject} {time}",
        "{topic}: {object} {adjective} sẽ được {action} {location}"
    ]
    
    # Tạo ngày ngẫu nhiên trong năm 2023
    def random_date():
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime("%d/%m/%Y")
    
    # Tạo tiêu đề ngẫu nhiên
    def create_random_title():
        template = random.choice(title_templates)
        
        # Tạo các thành phần của tiêu đề
        components = {
            "topic": random.choice(topics),
            "subject": random.choice(subjects),
            "action": random.choice(actions),
            "object": random.choice(objects),
            "adjective": random.choice(adjectives),
            "location": random.choice(locations),
            "time": random.choice(time_phrases),
            "connecting": random.choice(connecting_phrases),
            "result": random.choice(result_phrases),
            "date": random_date()
        }
        
        # Điền các thành phần vào mẫu
        title = template.format(**components)
        
        # Xác suất 30% thêm ngày tháng vào đầu tiêu đề
        if random.random() < 0.3:
            title = f"[{components['date']}] " + title
            
        return title
    
    # Tạo danh sách tiêu đề
    titles = set()  # Sử dụng set để đảm bảo không có tiêu đề trùng lặp
    while len(titles) < num_titles:
        titles.add(create_random_title())
    
    # Ghi tiêu đề vào tập tin
    with open(output_file, "w", encoding="utf-8") as file:
        for title in titles:
            file.write(title + "\n")
    
    print(f"Đã tạo {len(titles)} tiêu đề tin tức và lưu vào tập tin '{output_file}'")

if __name__ == "__main__":
    # Có thể tùy chỉnh số lượng tiêu đề và tên tập tin đầu ra
    num_titles = int(input("Nhập số lượng tiêu đề cần tạo (mặc định 1000): ") or 1000)
    output_file = input("Nhập tên file đầu ra (mặc định 'news_titles.txt'): ") or "news_titles.txt"
    
    generate_news_titles(num_titles, output_file)
