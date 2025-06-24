import unittest
import spacy
from analyzer.content_filter import ContentFilter
from analyzer.text_preprocessor import TextPreprocessor

class TestContentFilterWithRelaxedRules(unittest.TestCase):

    def setUp(self):
        """Set up the test case with the relaxed ContentFilter and a TextPreprocessor."""
        self.content_filter = ContentFilter()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading 'en_core_web_sm' spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.text_preprocessor = TextPreprocessor(self.nlp)
        self.job_description = """
            At Mindrift, innovation meets opportunity. We believe in using the power of collective intelligence to ethically shape the future of AI.
            What We Do
            The Mindrift platform connects specialists with AI projects from major tech innovators. Our mission is to unlock the potential of Generative AI by tapping into real-world expertise from across the globe.
            About The Role
            GenAI models are improving very quickly, and one of our goals is to make them capable of addressing specialized questions and achieving complex reasoning skills. If you join the platform as an AI Tutor in Coding, you'll have the opportunity to collaborate on these projects.
            Although every project is unique, you might typically:
            Generate prompts that challenge AI
            Define comprehensive scoring criteria to evaluate the accuracy of the AI's answers
            Correct the model's responses based on your domain-specific knowledge
            How To Get Started
            Simply apply to this post, qualify, and get the chance to contribute to projects aligned with your skills, on your own schedule. From creating training prompts to refining model responses, you'll help shape the future of AI while ensuring technology benefits everyone.
            Requirements
            You have a Bachelor's or Master's degree in Software Development, Computer Science, or other related fields. You have at least 3 years of professional experience with Rust. Code review experience is a must. Experience with AI projects is a plus. Your level of English is advanced (C1) or above. You are ready to learn new methods, able to switch between tasks and topics quickly and sometimes work with challenging, complex guidelines. Our freelance role is fully remote so, you just need a laptop, internet connection, time available and enthusiasm to take on a challenge.
            Benefits
            Why this freelance opportunity might be a great fit for you? Take part in a part-time, remote, freelance project that fits around your primary professional or academic commitments. Work on advanced AI projects and gain valuable experience that enhances your portfolio. Influence how future AI models understand and communicate in your field of expertise.
        """

    def test_relaxed_filter_extracts_real_job_requirements(self):
        """
        Confirms that the relaxed filter allows the text preprocessor to extract key 
        requirements from the complex, real-world job description.
        """
        chunks = self.text_preprocessor.improved_chunking_for_anomaly_detection(self.job_description)
        
        self.assertGreater(len(chunks), 0, "Chunking should now produce results.")
        
        full_chunk_text = " ".join(chunks)
        
        # Verify that essential requirements are present in the final chunks
        self.assertIn("Bachelor's or Master's degree", full_chunk_text, "Degree requirement should be extracted.")
        self.assertIn("experience with Rust", full_chunk_text, "Rust experience should be extracted.")
        self.assertIn("Code review experience is a must", full_chunk_text, "Code review requirement should be extracted.")
        self.assertIn("Experience with AI projects", full_chunk_text, "AI project experience should be extracted.")

    def test_filter_still_removes_explicit_exclusions(self):
        """
        Ensures the filter still removes chunks with clear, non-requirement-related
        phrases like application instructions or specific benefit statements.
        """
        chunks_to_test = [
            "This is a valid requirement chunk about skills.",
            "Please apply now to this position.",
            "We offer a competitive salary package.",
            "Contact us for more information.",
            "This role is focused on Python development."
        ]
        
        filtered_chunks = self.content_filter.filter_job_requirement_chunks(chunks_to_test)
        
        self.assertEqual(len(filtered_chunks), 2, "Should only keep the two valid requirement chunks.")
        self.assertIn("This is a valid requirement chunk about skills.", filtered_chunks)
        self.assertIn("This role is focused on Python development.", filtered_chunks)
        self.assertNotIn("Please apply now to this position.", filtered_chunks)

    def test_company_description_is_kept_if_it_lacks_hard_exclusions(self):
        """
        Tests that general company info, which no longer matches hard exclusion rules
        (like 'we are' or 'our mission'), is now kept.
        """
        company_description_chunk = "We believe in using the power of collective intelligence to ethically shape the future of AI."
        is_kept = self.content_filter._should_keep_chunk(company_description_chunk)
        self.assertTrue(is_kept, "General company philosophy should now be kept after relaxing the rules.")

if __name__ == '__main__':
    unittest.main() 