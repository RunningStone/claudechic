"""Unit tests for ChatApp methods."""

import base64

from claudechic import ChatApp


def test_image_attachment_message_building():
    """Test that images are correctly formatted in messages."""
    app = ChatApp()

    # Add a test image (path, filename, media_type, base64_data)
    test_data = base64.b64encode(b"fake image data").decode()
    app.pending_images.append(("/tmp/test.png", "test.png", "image/png", test_data))

    # Build message
    msg = app._build_message_with_images("What is this?")

    # Verify structure
    assert msg["type"] == "user"
    content = msg["message"]["content"]
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "What is this?"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"
    assert content[1]["source"]["data"] == test_data

    # Should clear pending images
    assert len(app.pending_images) == 0
