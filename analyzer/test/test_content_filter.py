import unittest
from analyzer.content_filter import ContentFilter

class TestContentFilter(unittest.TestCase):
    def setUp(self):
        self.content_filter = ContentFilter()

    def test_filter_with_only_requirements(self):
        chunks = [
            "Proficiency in Python programming.",
            "Experience with machine learning frameworks like TensorFlow or PyTorch.",
            "Strong understanding of data structures and algorithms."
        ]
        expected = chunks[:]
        result = self.content_filter.filter_job_requirement_chunks(chunks)
        self.assertEqual(result, expected)

    def test_filter_with_only_exclusions(self):
        chunks = [
            "About us: we are a leading innovator in the tech industry.",
            "We offer a competitive salary and benefits package.",
            "To apply, please submit your resume to our careers page.",
            "Our office is located in the heart of downtown.",
            "This is an exciting opportunity to join our team."
        ]
        expected = []
        result = self.content_filter.filter_job_requirement_chunks(chunks)
        self.assertEqual(result, expected)

    def test_filter_with_mixed_content(self):
        chunks = [
            "We are a fast-growing startup.",
            "Looking for a candidate with 3+ years of experience in Java.",
            "Our company values diversity and inclusion.",
            "Must be skilled in database management (SQL).",
            "We offer health insurance and a great pension plan.",
            "Please send your application to jobs@example.com."
        ]
        expected = [
            "Looking for a candidate with 3+ years of experience in Java.",
            "Must be skilled in database management (SQL)."
        ]
        result = self.content_filter.filter_job_requirement_chunks(chunks)
        self.assertEqual(result, expected)

    def test_filter_with_empty_list(self):
        chunks = []
        expected = []
        result = self.content_filter.filter_job_requirement_chunks(chunks)
        self.assertEqual(result, expected)

    def test_should_keep_chunk_with_borderline_cases(self):
        requirement_chunk = "The role requires strong leadership skills."
        non_requirement_chunk_company = "Our company is expanding its operations."
        
        self.assertTrue(self.content_filter._should_keep_chunk(requirement_chunk))
        self.assertFalse(self.content_filter._should_keep_chunk(non_requirement_chunk_company))

    def test_filter_with_short_chunks(self):
        # The current implementation keeps short chunks, as the length check is commented out.
        chunks = ["Python", "Java", "SQL"]
        expected = ["Python", "Java", "SQL"]
        result = self.content_filter.filter_job_requirement_chunks(chunks)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main() 