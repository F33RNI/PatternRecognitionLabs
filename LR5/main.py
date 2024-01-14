"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
"""
import os
from pathlib import Path

import google.generativeai as genai

# https://makersuite.google.com/app/apikey
API_KEY = "кучаСтранныхБукоф"

# Type your proxy here in format http://IP:port or http://user:password@IP:Port
# Example http://uuuser:password@111.222.123.200:1234
# SPECIFY HTTP EVEN IF IT IS HTTPS PROXY
# Or leave empty to disable proxy
PROXY = ""

# Text prompt
PROMPT = "What is on this photo:"

# Path to image to recognize
IMAGE_PATH = "путь/к/любой/картинковы/которую/хочите/распознац.jpg"

# Model name
MODEL = "gemini-pro-vision"


def main() -> None:
    # Configure proxy
    if PROXY:
        os.environ["http_proxy"] = PROXY

    # Configure google.generativeai with your api token
    genai.configure(api_key=API_KEY)

    # Load gemini-pro-vision model
    model = genai.GenerativeModel(MODEL)

    # Prepare image for contents
    prompt_picture = {"mime_type": "image/png", "data": Path(IMAGE_PATH).read_bytes()}

    # Ask model
    response = model.generate_content(contents=[PROMPT, prompt_picture])

    # Print response
    print(f"{MODEL} response: {response.text}")


if __name__ == "__main__":
    main()
