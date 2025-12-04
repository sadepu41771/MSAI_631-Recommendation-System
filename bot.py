from datetime import datetime
from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount

class MyBot(ActivityHandler):
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.

    async def on_message_activity(self, turn_context: TurnContext):
        text = turn_context.activity.text.strip()
        lower_text = text.lower()
        
        if lower_text in ["help", "capabilities", "what can you do?"]:
            await turn_context.send_activity(
                "I am an enhanced echo bot!\n\n"
                "Here is what I can do:\n"
                "1. **Time**: Ask 'time' to see the current server time.\n"
                "2. **Palindrome**: Type 'palindrome [word]' to check if it's a palindrome.\n"
                "3. **Reverse**: Send any other text and I'll say it backwards.\n"
                "4. **Error**: Type 'error' to test error handling."
            )
        elif lower_text == "time":
            current_time = datetime.now().strftime("%I:%M:%S %p")
            await turn_context.send_activity(f"The current server time is: {current_time}")
        elif lower_text.startswith("palindrome"):
            parts = text.split(maxsplit=1)
            if len(parts) > 1:
                word = parts[1]
                # Remove non-alphanumeric characters for a stricter palindrome check if desired, 
                # but for simplicity we'll just check the string provided.
                clean_word = ''.join(c.lower() for c in word if c.isalnum())
                is_palindrome = clean_word == clean_word[::-1]
                result = "is" if is_palindrome else "is not"
                await turn_context.send_activity(f"'{word}' {result} a palindrome (ignoring case/symbols).")
            else:
                await turn_context.send_activity("Please provide a word to check, e.g., 'palindrome racecar'.")
        elif lower_text == "error":
            # Simulate an error to demonstrate handling
            try:
                raise Exception("This is a simulated error for demonstration purposes.")
            except Exception as e:
                await turn_context.send_activity(f"Oops! I encountered an error: {str(e)}")
        else:
            # Reverse the text
            reversed_text = text[::-1]
            await turn_context.send_activity(f"You said '{text}', which backwards is '{reversed_text}'")

    async def on_members_added_activity(
        self,
        members_added: list[ChannelAccount],
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome! Type 'help' to see what I can do.")
