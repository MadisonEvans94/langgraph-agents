#!/usr/bin/env bash

# Test values
SUMMARY="The PS5 is Sony's latest gaming console, delivering cutting-edge graphics, lightning-fast load times, and deeply immersive experiences. It features advanced ray tracing, haptic feedback through the DualSense controller, 3D audio support, and an ultra-high-speed SSD that redefines performance. With its futuristic design and robust gaming library, the PS5 sets a new standard for next-gen entertainment."
IMAGE_URL="https://images.unsplash.com/photo-1607016284318-d1384bf5edd1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3NTQ5MTh8MHwxfHNlYXJjaHwxfHxQUzUlMjBnYW1pbmclMjBjb25zb2xlLnxlbnwwfHx8fDE3NDgyMDgxOTd8MA&ixlib=rb-4.1.0&q=80&w=1080"

# Endpoint and output file
ENDPOINT="http://localhost:8001/generate_html"
OUTPUT_FILE="html_agent_output.html"

# Call the API and write the HTML to a file
echo "Calling /generate_html..."
curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg summary "$SUMMARY" \
    --arg image_url "$IMAGE_URL" \
    '{summary: $summary, image_url: $image_url}')" \
  | jq -r '.html' > "$OUTPUT_FILE"

# Show result
echo "✅ HTML saved to $OUTPUT_FILE"
echo "🌐 Opening in browser..."
open "$OUTPUT_FILE"