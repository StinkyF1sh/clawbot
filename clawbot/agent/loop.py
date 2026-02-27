"""The main loop for the Clawbot agent."""


class ClawbotAgentLoop:
    """Main loop for the Clawbot agent."""

    def __init__(
        self,
        ):
        self.running = False

    def run(self):
        """Run the main loop."""
        self.running = True

        print(self.config)

    def stop(self):
        """Stop the main loop."""
        self.running = False
