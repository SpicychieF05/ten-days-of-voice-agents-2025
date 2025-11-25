import unittest
from unittest.mock import MagicMock
import json
import os
import sys

# Add the current directory to path so we can import agent
sys.path.append(os.path.dirname(__file__))

from agent import Assistant, VOICE_LEARN, VOICE_QUIZ, VOICE_TEACH_BACK

class MockSession:
    def __init__(self):
        self.tts = MagicMock()
        self.tts.voice = "initial_voice"

class TestActiveRecallTutor(unittest.TestCase):
    def setUp(self):
        self.session = MockSession()
        # We need to mock the content loading since the file might not be found in test env if paths differ,
        # but here we are in the same dir.
        self.agent = Assistant(self.session)

    def test_initial_state(self):
        self.assertEqual(self.agent.current_mode, "learn")
        self.assertEqual(self.agent.current_concept_id, "variables")
        self.assertEqual(self.session.tts.voice, VOICE_LEARN)

    def test_mode_switching(self):
        # Switch to Quiz
        self.agent.current_mode = "quiz"
        self.agent._update_voice()
        self.assertEqual(self.session.tts.voice, VOICE_QUIZ)

        # Switch to Teach-Back
        self.agent.current_mode = "teach_back"
        self.agent._update_voice()
        self.assertEqual(self.session.tts.voice, VOICE_TEACH_BACK)

        # Switch back to Learn
        self.agent.current_mode = "learn"
        self.agent._update_voice()
        self.assertEqual(self.session.tts.voice, VOICE_LEARN)

    def test_concept_switching(self):
        # Test valid concept
        self.agent.current_concept_id = "loops"
        self.assertEqual(self.agent.current_concept_id, "loops")

    def test_content_loading(self):
        self.assertTrue(len(self.agent.content) > 0)
        self.assertEqual(self.agent.content[0]['id'], "variables")

if __name__ == '__main__':
    unittest.main()
