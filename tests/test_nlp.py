import unittest
from chatbot.nlp import analyze_text


class TestNLP(unittest.TestCase):
    def test_greeting(self):
        result = analyze_text("Hello there!")
        self.assertEqual(result["intent"], "greeting")

    def test_positive_sentiment(self):
        result = analyze_text("I love this project, it is amazing!")
        self.assertEqual(result["sentiment"], "positive")

    def test_entities(self):
        result = analyze_text("Contact me at test@example.com")
        types = {e["type"] for e in result["entities"]}
        self.assertIn("EMAIL", types)


if __name__ == "__main__":
    unittest.main()
