const fs = require("fs");

// Đọc file JSON
fs.readFile("aa.json", "utf8", (err, data) => {
  if (err) {
    console.error("Lỗi khi đọc file:", err);
    return;
  }

  // Parse dữ liệu JSON
  const jsonData = JSON.parse(data);

  // Hàm để lấy ra các phần tử có title là "Question"
  function getQuestions(data) {
    return data.filter((item) => item.title === "Question");
  }

  // Hàm để lấy ra câu hỏi và đáp án đúng
  function getQuestionsWithCorrectAnswers(questions) {
    return questions.map((question) => {
      const correctAnswers = question._items
        .filter((item) => item._shouldBeSelected)
        .map((item) => item.text);

      return {
        question: question.body.replace(/<[^>]+>/g, ""), // Loại bỏ thẻ HTML
        correctAnswers: correctAnswers,
      };
    });
  }

  // Lấy các câu hỏi
  const questions = getQuestions(jsonData);

  // Lấy câu hỏi và đáp án đúng
  const questionsWithCorrectAnswers = getQuestionsWithCorrectAnswers(questions);

  // In ra kết quả
  console.log(questionsWithCorrectAnswers);
});
