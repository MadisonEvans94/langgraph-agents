#!/usr/bin/env bash
source .env
[ -z "$OPENAI_API_KEY" ] && { echo "Need OPENAI_API_KEY"; exit 1; }

IMAGE_PATH="reference.png"
OUTPUT_IMAGE="processor_ad_image.png"
PROMPT="Create a professional advertisement-style image showcasing this computer processor. Use dramatic lighting, sleek presentation, and highlight its technological sophistication. background must be blue"

resp=$(curl -s -X POST https://api.openai.com/v1/images/edits \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F model="gpt-image-1" \
  -F image="@${IMAGE_PATH}" \
  -F prompt="${PROMPT}" \
  -F n=1 \
  -F size="1024x1024")

echo "Full API response:"
echo "$resp" | jq .

echo "Decoding base64 image from response..."
echo "$resp" | jq -r '.data[0].b64_json' | base64 --decode > "$OUTPUT_IMAGE"

echo "Done — check $OUTPUT_IMAGE"