# -*- coding: utf-8 -*-
"""
Phân tích cảm xúc trên bộ dữ liệu IMDB sử dụng các mô hình Deep Learning
So sánh hiệu suất của CNN, LSTM và Transformer

# Giải thích các chỉ số đánh giá:

## Accuracy (Độ chính xác)
- Đo lường tỷ lệ dự đoán đúng trên tổng số mẫu
- Công thức: (TP + TN) / (TP + TN + FP + FN)
- Ưu điểm: Dễ hiểu, trực quan
- Nhược điểm: Không phù hợp với dữ liệu mất cân bằng

## Precision (Độ chính xác dương tính)
- Đo lường tỷ lệ dự đoán đúng trong số các mẫu được dự đoán là tích cực
- Công thức: TP / (TP + FP)
- Ý nghĩa: Trong số những đánh giá mà mô hình dự đoán là tích cực, có bao nhiêu thực sự là tích cực
- Sử dụng khi chi phí của false positive cao (ví dụ: gửi email quảng cáo cho người không quan tâm)

## Recall (Độ nhạy)
- Đo lường tỷ lệ mẫu tích cực thực sự được phát hiện
- Công thức: TP / (TP + FN)
- Ý nghĩa: Trong số những đánh giá thực sự tích cực, mô hình phát hiện được bao nhiêu
- Sử dụng khi chi phí của false negative cao (ví dụ: không phát hiện bệnh nhân mắc bệnh)

## F1 Score
- Trung bình điều hòa (harmonic mean) của Precision và Recall
- Công thức: 2 * (Precision * Recall) / (Precision + Recall)
- Ý nghĩa: Cân bằng giữa Precision và Recall
- Sử dụng khi cần cân nhắc cả false positive và false negative

## Confusion Matrix (Ma trận nhầm lẫn)
- Bảng thể hiện số lượng dự đoán đúng và sai cho mỗi lớp
- Thành phần:
  * True Positive (TP): Dự đoán đúng mẫu tích cực
  * True Negative (TN): Dự đoán đúng mẫu tiêu cực
  * False Positive (FP): Dự đoán sai mẫu tiêu cực thành tích cực
  * False Negative (FN): Dự đoán sai mẫu tích cực thành tiêu cực
- Giúp phân tích chi tiết hiệu suất của mô hình
"""


# 2. Chuẩn bị dữ liệu
# Giới hạn số lượng từ vựng sử dụng
max_features = 10000  
# Tải dữ liệu IMDB
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

print(f"Số lượng mẫu huấn luyện: {len(X_train)}")
print(f"Số lượng mẫu kiểm tra: {len(X_test)}")

# Kiểm tra một vài mẫu dữ liệu
print(f"Mẫu đầu tiên (dạng số): {X_train[0][:10]}...")
print(f"Nhãn của mẫu đầu tiên: {y_train[0]}")

# Lấy từ điển ánh xạ từ index sang từ
word_index = imdb.get_word_index()
# Đảo ngược từ điển để ánh xạ từ index sang từ
reverse_word_index = {value: key for key, value in word_index.items()}

# Hàm giải mã đánh giá từ dạng số về dạng văn bản


"""
# Phân tích dữ liệu IMDB

Bộ dữ liệu IMDB chứa 50,000 bài đánh giá phim, 
được chia thành:
- 25,000 mẫu huấn luyện
- 25,000 mẫu kiểm tra

Mỗi bài đánh giá được gán nhãn:
- 0: Tiêu cực
- 1: Tích cực

Dữ liệu đã được tiền xử lý và chuyển 
thành chuỗi số, mỗi số đại diện cho một từ trong từ điển.
"""

"""
# Tiền xử lý dữ liệu

## Padding
Các bài đánh giá có độ dài khác nhau, 
cần chuẩn hóa để đưa vào mô hình:
- Cắt bớt các bài đánh giá quá dài
- Thêm padding (giá trị 0) vào các bài đánh giá quá ngắn
- Độ dài chuẩn được chọn là 200 từ, 
dựa trên phân tích phân bố độ dài
"""

# Chuẩn hóa độ dài các bài đánh giá
maxlen = 200  # Độ dài tối đa của mỗi bài đánh giá

# Padding các chuỗi để có cùng độ dài
X_train_pad = pad_sequences(X_train, maxlen=maxlen)
X_test_pad = pad_sequences(X_test, maxlen=maxlen)

print(f"Shape của X_train sau khi padding: {X_train_pad.shape}")
print(f"Shape của X_test sau khi padding: {X_test_pad.shape}")

# Kích thước vector embedding
embedding_dim = 128

"""
# Mô hình CNN cho phân tích cảm xúc
Mô hình CNN trong mã của bạn là Mạng nơ-ron tích chập 1D được thiết kế để phân loại văn bản

## Kiến trúc mô hình CNN
- Sequential: Mô hình dòng chảy (mạng nơ-ron tuần tự)
- Lớp Embedding: Chuyển đổi các chỉ số từ thành 
vector đặc trưng
- Lớp Dropout: Giảm overfitting - 
overfitting là hiện tượng mô hình học quá mức dữ liệu huấn luyện, dẫn đến hiệu suất trên tập kiểm tra giảm
- Các lớp Conv1D: 
Trích xuất đặc trưng cục bộ với các
 kích thước cửa sổ khác nhau
-  MaxPooling và GlobalMaxPooling: Giảm kích thước và lấy đặc trưng quan trọng nhất
- Các lớp Dense: Phân loại dựa trên đặc trưng đã trích xuất
- Các lớp fully connected: Phân loại dựa trên đặc trưng đã trích xuất
- Lớp output với sigmoid cho phân loại nhị phân
## Ưu điểm của CNN trong xử lý văn bản
- Phát hiện các mẫu cục bộ (n-gram) trong văn bản
- Hiệu quả tính toán cao hơn so với RNN
- Có thể xử lý song song
"""


"""
# Mô hình LSTM cho phân tích cảm xúc

## Kiến trúc mô hình LSTM
- Lớp Embedding: Chuyển đổi các chỉ số từ thành vector đặc trưng
- Bidirectional LSTM: Xử lý chuỗi theo cả hai hướng để nắm bắt ngữ cảnh tốt hơn
- Các lớp Dense: Phân loại dựa trên đặc trưng đã trích xuất
- Dropout: Giảm overfitting

## Ưu điểm của LSTM trong xử lý văn bản
- Nắm bắt được phụ thuộc dài hạn trong văn bản
- Hiểu được ngữ cảnh và thứ tự của các từ
- Xử lý tốt các chuỗi có độ dài khác nhau
"""

# 6. Xây dựng mô hình RNN (LSTM)
def build_lstm_model():
    model_lstm = Sequential([
        # Lớp Embedding
        Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen),
        
        # Lớp LSTM hai chiều đầu tiên, trả về chuỗi để kết nối với LSTM tiếp theo
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        
        # Lớp LSTM hai chiều thứ hai
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        
        # Lớp fully connected
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        # Lớp output
        Dense(1, activation='sigmoid')
    ])

    # Biên dịch mô hình
    model_lstm.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model_lstm


"""
# Mô hình Transformer cho phân tích cảm xúc

## Kiến trúc mô hình Transformer
- Lớp Embedding + Positional Encoding: Chuyển đổi các chỉ số từ thành vector đặc trưng và thêm thông tin vị trí
- Multi-Head Attention: Cho phép mô hình tập trung vào các phần khác nhau của chuỗi đầu vào
- Feed-Forward Network: Xử lý thông tin từ cơ chế attention
- Layer Normalization và Residual Connection: Giúp huấn luyện ổn định
- Global Average Pooling: Tổng hợp thông tin từ toàn bộ chuỗi
- Các lớp Dense: Phân loại dựa trên đặc trưng đã trích xuất

## Ưu điểm của Transformer trong xử lý văn bản
- Xử lý song song toàn bộ chuỗi đầu vào
- Nắm bắt được mối quan hệ giữa các từ ở xa nhau
- Hiệu quả với các văn bản dài
- Khả năng mở rộng tốt
"""

# 9. Xây dựng mô hình Transformer

# Lớp MultiHeadAttention tùy chỉnh
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads  # Số lượng đầu attention
        self.d_model = d_model  # Kích thước của mô hình
        
        # Đảm bảo d_model chia hết cho num_heads
        assert d_model % self.num_heads == 0
        
        # Kích thước của mỗi đầu attention
        self.depth = d_model // self.num_heads
        
        # Các lớp Dense để tạo query, key, value
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        # Lớp Dense cuối cùng
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        # Chia tensor thành nhiều đầu
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # Chuyển vị để có shape (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        # Tạo query, key, value thông qua các lớp Dense
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # Chia thành nhiều đầu
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Tính toán scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Thêm mask nếu được cung cấp
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
            
        # Softmax để có trọng số attention
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Áp dụng trọng số attention vào value
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        
        # Ghép các đầu lại
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Lớp Dense cuối cùng
        output = self.dense(concat_attention)
        
        return output

# Lớp TransformerBlock
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.mha = MultiHeadAttention(d_model, num_heads)
        
        # Feed forward network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        # Layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, x, training=True):
        # Multi-head attention với residual connection và layer normalization
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward network với residual connection và layer normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

# Hàm tạo positional encoding
def get_angles(pos, i, d_model):
    # Tính góc cho positional encoding
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    # Tạo positional encoding
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)
    
    # Áp dụng sin cho các chỉ số chẵn
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Áp dụng cos cho các chỉ số lẻ
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

# Hàm xây dựng mô hình Transformer
def build_transformer_model(max_features, maxlen, embedding_dim=128, num_heads=8, ff_dim=512):
    inputs = Input(shape=(maxlen,))
    
    # Lớp Embedding
    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dim)(inputs)
    
    # Thêm positional encoding
    pos_encoding = positional_encoding(maxlen, embedding_dim)
    x = embedding_layer + pos_encoding[:, :maxlen, :]
    
    # Dropout
    x = Dropout(0.1)(x)
    
    # Các khối Transformer
    transformer_block1 = TransformerBlock(embedding_dim, num_heads, ff_dim)
    transformer_block2 = TransformerBlock(embedding_dim, num_heads, ff_dim)
    
    x = transformer_block1(x)
    x = transformer_block2(x)
    
    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    
    # Các lớp Dense
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Khởi tạo mô hình Transformer
model_transformer = build_transformer_model(max_features, maxlen)
model_transformer.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model_transformer.summary()

# 10. Huấn luyện mô hình Transformer
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('./checkpoint/best_transformer_model.keras', save_best_only=True, monitor='val_accuracy')

history_transformer = model_transformer.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=64,  # Batch size nhỏ hơn do mô hình phức tạp hơn
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# 11. Đánh giá mô hình Transformer
loss_transformer, accuracy_transformer = model_transformer.evaluate(X_test_pad, y_test)
print(f"Độ chính xác trên tập kiểm tra của mô hình Transformer: {accuracy_transformer:.4f}")

# Dự đoán trên tập test
y_pred_transformer = (model_transformer.predict(X_test_pad) > 0.5).astype(int).flatten()
# Tính toán các chỉ số đánh giá
precision_transformer = precision_score(y_test, y_pred_transformer)
recall_transformer = recall_score(y_test, y_pred_transformer)
accuracy_transformer = accuracy_score(y_test, y_pred_transformer)
f1_transformer = f1_score(y_test, y_pred_transformer)

print(f"Transformer Model Evaluation:")
print(f"Precision: {precision_transformer:.4f}")
print(f"Recall: {recall_transformer:.4f}")
print(f"Accuracy: {accuracy_transformer:.4f}")
print(f"F1: {f1_transformer:.4f}")

# Vẽ confusion matrix cho mô hình Transformer
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 3)
cm_transformer = confusion_matrix(y_test, y_pred_transformer)
sns.heatmap(cm_transformer, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Transformer')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""
# Phân tích kết quả Confusion Matrix

Confusion Matrix là một công cụ trực quan để đánh giá hiệu suất của mô hình phân loại:

- Hàng: Giá trị thực tế
- Cột: Giá trị dự đoán

Trong bài toán phân loại nhị phân:
- True Positive (TP): Góc trên bên phải - Dự đoán đúng mẫu tích cực
- True Negative (TN): Góc trên bên trái - Dự đoán đúng mẫu tiêu cực
- False Positive (FP): Góc dưới bên phải - Dự đoán sai mẫu tiêu cực thành tích cực
- False Negative (FN): Góc dưới bên trái - Dự đoán sai mẫu tích cực thành tiêu cực

Từ Confusion Matrix, ta có thể tính toán:
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
"""

# Vẽ biểu đồ độ chính xác trên tập validation
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(history_transformer.history['val_accuracy'], label='Transformer')
plt.title('Độ chính xác trên tập validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""
# So sánh các mô hình Deep Learning

## CNN (Convolutional Neural Network)
- Ưu điểm: Nhanh, hiệu quả trong việc phát hiện các mẫu cục bộ
- Nhược điểm: Khó nắm bắt các phụ thuộc dài hạn

## LSTM (Long Short-Term Memory)
- Ưu điểm: Nắm bắt tốt ngữ cảnh và phụ thuộc dài hạn
- Nhược điểm: Chậm hơn CNN, khó huấn luyện với chuỗi dài

## Transformer
- Ưu điểm: Xử lý song song, nắm bắt tốt mối quan hệ giữa các từ ở xa nhau
- Nhược điểm: Phức tạp hơn, yêu cầu nhiều dữ liệu hơn để huấn luyện hiệu quả
"""

# 12. So sánh các mô hình CNN, LSTM, Transformer
# Tạo bảng so sánh 3 mô hình
models = ['LSTM', 'CNN', 'Transformer']
predictions = [y_pred_lstm, y_pred_cnn, y_pred_transformer]

results = []
for model_name, y_pred in zip(models, predictions):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

# Hiển thị kết quả
results_df = pd.DataFrame(results)
results_df = results_df.set_index('Model')
print(results_df)

"""
# Phân tích kết quả so sánh các mô hình

Khi so sánh các mô hình, cần xem xét nhiều chỉ số khác nhau:

## Accuracy (Độ chính xác)
- Đo lường tổng thể hiệu suất của mô hình
- Phù hợp khi dữ liệu cân bằng giữa các lớp

## Precision (Độ chính xác dương tính)
- Quan trọng khi chi phí của false positive cao
- Ví dụ: Trong hệ thống lọc spam, gán nhãn email quan trọng là spam sẽ gây hậu quả nghiêm trọng

## Recall (Độ nhạy)
- Quan trọng khi chi phí của false negative cao
- Ví dụ: Trong chẩn đoán y tế, bỏ sót bệnh nhân mắc bệnh sẽ gây hậu quả nghiêm trọng

## F1 Score
- Cân bằng giữa precision và recall
- Phù hợp khi cần cân nhắc cả false positive và false negative
- Đặc biệt hữu ích khi dữ liệu mất cân bằng

Việc lựa chọn mô hình tốt nhất phụ thuộc vào yêu cầu cụ thể của ứng dụng.
"""

# 13. Dự đoán với các bài đánh giá mới
def prepare_text_for_prediction(text, word_index, maxlen=200):
    # Chuyển đổi văn bản thành chuỗi số
    words = text.lower().split()
    
    # Chỉ lấy các từ có trong từ điển
    sequence = []
    for word in words:
        # word_index bắt đầu từ 1, và trong dataset IMDb có offset +3
        if word in word_index:
            sequence.append(word_index[word] + 3)
        else:
            sequence.append(3)  # 3 là chỉ số cho từ không biết
    
    # Kiểm tra nếu chuỗi rỗng
    if not sequence:
        sequence = [3]  # Thêm ít nhất một token nếu chuỗi rỗng
    
    # Padding chuỗi
    padded_sequence = pad_sequences([sequence], maxlen=maxlen)
    
    return padded_sequence

"""
# Ứng dụng thực tế của mô hình phân tích cảm xúc

Các mô hình phân tích cảm xúc có nhiều ứng dụng trong thực tế:

1. Phân tích phản hồi khách hàng
   - Hiểu cảm xúc của khách hàng về sản phẩm/dịch vụ
   - Phát hiện vấn đề cần cải thiện

2. Giám sát thương hiệu trên mạng xã hội
   - Theo dõi nhận thức của công chúng về thương hiệu
   - Phát hiện khủng hoảng truyền thông tiềm ẩn

3. Phân tích thị trường và đối thủ cạnh tranh
   - Hiểu phản ứng của thị trường đối với sản phẩm mới
   - So sánh cảm xúc về sản phẩm của mình với đối thủ

4. Cải thiện trải nghiệm người dùng
   - Phân tích phản hồi để cải thiện giao diện người dùng
   - Tối ưu hóa trải nghiệm khách hàng

5. Hỗ trợ ra quyết định
   - Cung cấp thông tin cho các quyết định kinh doanh
   - Dự đoán xu hướng thị trường dựa trên cảm xúc
"""

# Các bài đánh giá mẫu để kiểm tra
sample_reviews = [
    "This movie was fantastic! I really enjoyed every moment of it. The acting was superb and the plot was engaging.",
    "Terrible film. Complete waste of time and money. The acting was wooden and the plot made no sense at all.",
    "The movie was okay, not great but not terrible either. Some parts were good but overall it was just average."
]

# Dự đoán với 3 mô hình
for review in sample_reviews:
    # Chuẩn bị văn bản
    prepared_review = prepare_text_for_prediction(review, word_index, maxlen)
    
    # Dự đoán với từng mô hình
    pred_lstm = model_lstm.predict(prepared_review)[0][0]
    pred_cnn = model_cnn.predict(prepared_review)[0][0]
    pred_transformer = model_transformer.predict(prepared_review)[0][0]
    
    print(f"Đánh giá: {review}")
    print(f"LSTM: {'Tích cực' if pred_lstm > 0.5 else 'Tiêu cực'} (Điểm: {pred_lstm:.4f})")
    print(f"CNN: {'Tích cực' if pred_cnn > 0.5 else 'Tiêu cực'} (Điểm: {pred_cnn:.4f})")
    print(f"Transformer: {'Tích cực' if pred_transformer > 0.5 else 'Tiêu cực'} (Điểm: {pred_transformer:.4f})")
    print("-" * 100)

"""
# Kết luận và hướng phát triển

## Kết luận
- Các mô hình deep learning như CNN, LSTM và Transformer đều hiệu quả trong phân tích cảm xúc
- Mỗi mô hình có ưu và nhược điểm riêng, phù hợp với các tình huống khác nhau
- Việc lựa chọn mô hình phụ thuộc vào yêu cầu cụ thể của ứng dụng

## Hướng phát triển
1. Cải thiện tiền xử lý dữ liệu
   - Sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên nâng cao
   - Áp dụng các phương pháp làm sạch dữ liệu tốt hơn

2. Thử nghiệm với các kiến trúc mô hình khác
   - BERT, GPT, RoBERTa và các mô hình ngôn ngữ tiền huấn luyện khác
   - Kết hợp các mô hình để tận dụng ưu điểm của từng loại

3. Mở rộng phạm vi phân tích
   - Phân loại đa lớp (rất tiêu cực, tiêu cực, trung tính, tích cực, rất tích cực)
   - Phân tích cảm xúc dựa trên khía cạnh (aspect-based sentiment analysis)

4. Áp dụng cho các ngôn ngữ khác
   - Phát triển mô hình đa ngôn ngữ
   - Thích ứng với đặc thù của từng ngôn ngữ
"""
