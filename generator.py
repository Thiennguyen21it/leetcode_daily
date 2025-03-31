"""
 Trong Python, Generator là một hàm đặc biệt cho phép bạn tạo ra một chuỗi các giá trị liên tục theo nhu cầu (lazy evaluation), mà không cần lưu toàn bộ chuỗi đó trong bộ nhớ.

Generator rất tiện lợi khi làm việc với các tập dữ liệu lớn, vì nó giúp tiết kiệm bộ nhớ và tăng hiệu suất.
"""
def my_generator():
    yield 1
    yield 2
    yield 3

for num in my_generator():
    print(num)
 
""" 
Generator hoạt động thế nào?

    Khi gọi generator, hàm không thực thi ngay lập tức, mà chỉ tạo một đối tượng generator.

    Mỗi khi bạn lặp qua generator (dùng vòng lặp for, hoặc hàm next()), hàm mới thực thi đến câu lệnh yield tiếp theo và trả về giá trị.
"""    
def gen_numbers():
    print("start")
    yield 10
    print("Continue")
    yield 20
    print("End")

gen = gen_numbers()

